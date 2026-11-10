"""Write the latest.json the updater reads, from a built installer and its .sig file.

    python packaging/make_latest_json.py --setup <setup.exe> --url <its download URL> --out latest.json

The version comes from tauri.conf.json unless given, the notes from that version's CHANGELOG section.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def changelog_notes(version: str, path: Path = ROOT / "CHANGELOG.md") -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, flags=re.M | re.S)
    return match.group(1).strip() if match else ""


def build(setup: Path, url: str, version: str, notes: str) -> dict:
    signature = Path(f"{setup}.sig")
    if not signature.is_file():
        raise SystemExit(f"{signature} not found: build with createUpdaterArtifacts and the signing key")
    return {
        "version": version,
        "notes": notes,
        "pub_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platforms": {"windows-x86_64": {"signature": signature.read_text(encoding="utf-8").strip(), "url": url}},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, default=Path("latest.json"))
    parser.add_argument("--version", default="")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args()
    version = args.version or json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))["version"]
    notes = changelog_notes(version) if args.notes is None else args.notes
    args.out.write_text(json.dumps(build(args.setup, args.url, version, notes), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.out}: {version}")


if __name__ == "__main__":
    main()
