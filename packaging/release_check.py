"""Before a release build: the tag, the version in the files and the CHANGELOG section must agree.

    python packaging/release_check.py --tag v1.2.0 --notes-out notes.md [--github-output <file>]

Run by .github/workflows/release.yml as its first step, so a wrong tag fails in seconds instead
of after the tests and the build. Without --tag (a build from a branch) only the files are
checked. Writes this version's CHANGELOG section to --notes-out: the draft release's notes.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TAG = re.compile(r"^v(\d+\.\d+\.\d+)$")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(tag: str, root: Path | None = None) -> tuple[str, str, list[str]]:
    """(version, notes, problems). An empty problem list means the release can go ahead."""
    root = root or ROOT
    bump = _load("bump_version", root / "scripts" / "bump_version.py")
    latest = _load("make_latest_json", root / "packaging" / "make_latest_json.py")
    bump.ROOT, bump.CHANGELOG = root, root / "CHANGELOG.md"
    problems = list(bump.problems())
    found = {version for versions in bump.read_versions().values() for version in versions}
    version = next(iter(found)) if len(found) == 1 else ""
    if tag:
        match = TAG.match(tag)
        if not match:
            problems.append(f"tag {tag} is not of the form v1.2.0")
        elif version and match.group(1) != version:
            problems.append(
                f"tag {tag} but the files say {version}: run scripts/bump_version.py {match.group(1)}, "
                f"commit, and tag that commit"
            )
    notes = latest.changelog_notes(version, root / "CHANGELOG.md") if version else ""
    if version and not notes:
        problems.append(f"the CHANGELOG.md section for {version} is empty: it becomes the release notes")
    return version, notes, problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tag", default="", help="the pushed tag, e.g. v1.2.0; empty for a build from a branch")
    parser.add_argument("--notes-out", type=Path)
    parser.add_argument("--github-output", type=Path, help="append version= to this file ($GITHUB_OUTPUT)")
    args = parser.parse_args(argv)
    version, notes, problems = check(args.tag)
    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        return 1
    if args.notes_out:
        args.notes_out.parent.mkdir(parents=True, exist_ok=True)
        args.notes_out.write_text(notes + "\n", encoding="utf-8")
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as out:
            out.write(f"version={version}\n")
    print(f"{args.tag or 'branch build'}: version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
