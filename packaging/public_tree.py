"""Export the public source tree (github.com/NG666666158/xunxun) from this private repository.

    python packaging/public_tree.py --out <checkout of the public repo> [--ref HEAD]

The public repository holds the app, its tests and the evaluation framework, nothing else:

* ``evaluation/`` keeps only ``evaluation/suite`` (the framework and the CI slice). The other
  folders hold results on the owner's own documents, and questions written from copyrighted
  books; they stay private.
* Files listed in EXCLUDE are left out (development logs, the old web-demo screenshot, the
  private term list itself).
* Every text file is checked against ``packaging/private_terms.txt`` (names of the owner's
  company, people, products and files; one term per line, kept in this repository only). A hit
  stops the export and lists the files, so nothing private is published by accident.

The folder given by --out is emptied except for its ``.git`` and refilled; commit and push there.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
TERMS = ROOT / "packaging" / "private_terms.txt"
KEEP_EVALUATION = ("evaluation/suite/",)
EXCLUDE = {
    "packaging/private_terms.txt",
    "assets/ui-preview.png",
}
EXCLUDE_PREFIXES = ("data/logs/",)
BINARY = {".png", ".jpg", ".jpeg", ".ico", ".icns", ".npz", ".woff", ".woff2", ".ttf", ".exe", ".dll", ".pdf",
          ".docx", ".xlsx", ".pptx", ".sqlite3", ".bin", ".wasm", ".gz", ".zip"}


def published(name: str) -> bool:
    if name in EXCLUDE or name.startswith(EXCLUDE_PREFIXES) or "/__pycache__/" in name:
        return False
    if name.startswith("evaluation/"):
        return name.startswith(KEEP_EVALUATION)
    return True


def private_terms() -> list[str]:
    if not TERMS.exists():
        return []
    lines = (line.strip() for line in TERMS.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    out = args.out.resolve()
    if ROOT == out or ROOT in out.parents:
        raise SystemExit("--out must be outside this repository")
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar", args.ref], capture_output=True, check=True).stdout
    terms = private_terms()
    hits: dict[str, list[str]] = {}
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if not member.isfile() or not published(member.name):
                continue
            data = tar.extractfile(member).read()
            files[member.name] = data
            if Path(member.name).suffix.lower() in BINARY:
                continue
            text = data.decode("utf-8", "ignore")
            found = [term for term in terms if term in text]
            if found:
                hits[member.name] = found
    if hits:
        for name, found in sorted(hits.items()):
            print(f"private: {name}: {', '.join(found[:5])}", file=sys.stderr)
        raise SystemExit(f"{len(hits)} files hold private terms; nothing was written")
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():
        if child.name == ".git":
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for name, data in files.items():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    print(f"{len(files)} files written to {out} ({len(terms)} private terms checked)")


if __name__ == "__main__":
    main()
