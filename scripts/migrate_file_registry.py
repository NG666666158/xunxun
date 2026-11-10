"""Move documents indexed before the file registry onto registry file ids.

Older versions keyed chunks by a hash of file name + content. This script gives each
such document a random file_id, re-keys its chunks, parents and parsed snapshot in place
(no re-embedding), and records it in the file registry. The original is looked for in
``data/uploads/<old id>/`` and, by name + legacy hash, in ``--search-dir`` folders. With
``--watch`` those folders also become watched folders, so later edits are picked up.

The app runs the same migration (without search folders) on startup; documents whose
original cannot be found are marked missing and can be relocated from the UI. Re-running
this script with a search folder locates them afterwards.

    python scripts/migrate_file_registry.py --search-dir sample_data/generated --watch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "project"))

from core.rag_system import RAGSystem  # noqa: E402
from library import LibraryError  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="把旧文档迁移到本地文件表")
    parser.add_argument("--search-dir", action="append", type=Path, default=[], help="查找原文件的目录，可重复")
    parser.add_argument("--watch", action="store_true", help="同时把这些目录加为监控目录")
    args = parser.parse_args()

    system = RAGSystem().initialize()
    library = system.library
    search_dirs = [path.resolve() for path in args.search_dir]
    if args.watch:
        for directory in search_dirs:
            try:
                root = library.add_root(str(directory), background=False)
                print(f"监控目录：{root['path']}")
            except LibraryError as exc:
                print(f"跳过监控 {directory}：{exc}", file=sys.stderr)
    for record in library.migrate_legacy(search_dirs):
        print(json.dumps({key: record[key] for key in ("file_id", "legacy_doc_id", "name", "status", "path")}, ensure_ascii=False))
    if args.watch:
        # Files in the new folders that were not legacy documents still need indexing.
        for outcome in library.run_until_idle():
            print(f"index: {outcome}")
    print(json.dumps(library.registry.status_counts(), ensure_ascii=False))


if __name__ == "__main__":
    main()
