"""Download the public datasets listed in sources.json into evaluation/external/<dataset>/.

Every file comes from a pinned revision and is checked before it is kept: the sha256
of a Hugging Face LFS file, or the git blob sha1 of a plain git file (for GitHub
files, the blob sha1 the pinned commit's tree lists). Nothing downloaded is executed;
the adapters only parse JSON / parquet / text.

Hugging Face files are fetched from a mirror (EVAL_HF_ENDPOINT, default
https://hf-mirror.com) without the system proxy, which is much faster from this network;
set EVAL_HF_ENDPOINT=https://huggingface.co to use the origin. GitHub uses the proxy
from the environment, if any.

Each dataset folder gets FETCHED.json: URLs, revision, size, sha256 per file, the
license note and one checksum over all files (recorded in every result file).

Usage (run with -I, the data is untrusted):
  python -I evaluation/suite/fetch.py [dataset ...]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTERNAL = Path(os.environ.get("EVAL_EXTERNAL_DIR", ROOT / "evaluation" / "external"))
SOURCES = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
HF_ENDPOINT = os.environ.get("EVAL_HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
USER_AGENT = "rag-eval-suite/1.0"


def _opener(direct: bool):
    handlers = [urllib.request.ProxyHandler({})] if direct else []
    return urllib.request.build_opener(*handlers)


def _get(url: str, direct: bool, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with _opener(direct).open(request, timeout=timeout) as response:
        return response.read()


def _download(url: str, target: Path, direct: bool, attempts: int = 5) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with _opener(direct).open(request, timeout=300) as response, partial.open("wb") as handle:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
            partial.replace(target)
            return
        except Exception as exc:  # network hiccups: retry, then give up loudly
            if attempt == attempts - 1:
                raise RuntimeError(f"download failed: {url}: {exc}") from exc
            time.sleep(2 ** attempt * 3)


def file_digests(path: Path) -> tuple[str, str, int]:
    """(sha256, git blob sha1, size)."""
    size = path.stat().st_size
    sha256, blob = hashlib.sha256(), hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest(), size


def _github_tree(repo: str, revision: str) -> dict[str, dict]:
    payload = json.loads(_get(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1", direct=False))
    if payload.get("truncated"):
        raise RuntimeError(f"tree listing of {repo}@{revision} is truncated")
    return {item["path"]: item for item in payload["tree"] if item["type"] == "blob"}


def plan(name: str) -> list[dict]:
    """Concrete files (url, target, expected digests) of one dataset."""
    items = []
    trees: dict[tuple[str, str], dict] = {}
    for spec in SOURCES[name]["files"]:
        if spec["host"] == "hf":
            url = f"{HF_ENDPOINT}/datasets/{spec['repo']}/resolve/{spec['revision']}/{spec['path']}"
            origin = f"https://huggingface.co/datasets/{spec['repo']}/blob/{spec['revision']}/{spec['path']}"
            items.append({**spec, "url": url, "origin": origin, "target": spec["save_as"], "direct": "hf-mirror.com" in HF_ENDPOINT})
            continue
        key = (spec["repo"], spec["revision"])
        if key not in trees:
            trees[key] = _github_tree(*key)
        tree = trees[key]
        paths = [spec["path"]] if "path" in spec else sorted(p for p in tree if p.startswith(spec["tree_prefix"]))
        for path in paths:
            entry = tree[path]
            target = spec.get("save_as") if "path" in spec else f"{spec['save_dir']}/{Path(path).name}"
            url = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['revision']}/{path}"
            items.append({"host": "github", "repo": spec["repo"], "revision": spec["revision"], "path": path,
                          "url": url, "origin": url, "target": target, "size": entry["size"],
                          "git_sha1": entry["sha"], "direct": False})
    return items


def verify(item: dict, path: Path) -> dict:
    sha256, blob, size = file_digests(path)
    problems = []
    if item.get("size") is not None and size != item["size"]:
        problems.append(f"size {size} != {item['size']}")
    if item.get("sha256") and sha256 != item["sha256"]:
        problems.append("sha256 mismatch")
    if item.get("git_sha1") and blob != item["git_sha1"]:
        problems.append("git blob sha1 mismatch")
    return {"path": item["target"], "source_path": item["path"], "url": item["origin"], "revision": item["revision"],
            "size": size, "sha256": sha256, "problems": problems}


def fetch(name: str, workers: int = 4) -> dict:
    folder = EXTERNAL / name
    items = plan(name)

    def one(item: dict) -> dict:
        target = folder / item["target"]
        if target.is_file():
            record = verify(item, target)
            if not record["problems"]:
                return record
        _download(item["url"], target, item["direct"])
        record = verify(item, target)
        if record["problems"]:
            target.rename(target.with_name(target.name + ".rejected"))
            raise RuntimeError(f"{name}/{item['target']}: {'; '.join(record['problems'])}")
        return record

    with ThreadPoolExecutor(workers) as pool:
        records = list(pool.map(one, items))
    for record in records:
        record.pop("problems")
    checksum = hashlib.sha256("".join(f"{r['path']}\t{r['sha256']}\n" for r in sorted(records, key=lambda r: r["path"])).encode()).hexdigest()
    meta = {key: SOURCES[name][key] for key in ("homepage", "upstream", "license")}
    manifest = {"dataset": name, **meta, "checksum": checksum, "files": records,
                "total_bytes": sum(r["size"] for r in records), "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (folder / "FETCHED.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=[k for k in SOURCES if not k.startswith("_")])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for name in args.datasets:
        manifest = fetch(name, args.workers)
        print(f"{name}: {len(manifest['files'])} files, {manifest['total_bytes'] / 1e6:.1f} MB, checksum {manifest['checksum'][:16]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
