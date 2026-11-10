"""Set the app's version everywhere it is written down, and open its CHANGELOG section.

    python scripts/bump_version.py 1.2.0

Without an argument it only checks that every file agrees (project/tests/test_release.py runs
the same check). The version is what makes an in-place upgrade or an update possible: the
installer and the updater compare it, and they refuse to go backwards.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
CHANGELOG = ROOT / "CHANGELOG.md"
VERSION = re.compile(r"^\d+\.\d+\.\d+$")


def _json_version(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    found = [data["version"]]
    # package-lock.json repeats it for the root package.
    if "packages" in data:
        found.append(data["packages"][""]["version"])
    return found


def read_versions() -> dict[str, list[str]]:
    """Every place the version is written, by file."""
    config = (ROOT / "project/config.py").read_text(encoding="utf-8")
    cargo = (ROOT / "desktop/src-tauri/Cargo.toml").read_text(encoding="utf-8")
    lock = (ROOT / "desktop/src-tauri/Cargo.lock").read_text(encoding="utf-8")
    return {
        "desktop/src-tauri/tauri.conf.json": _json_version(ROOT / "desktop/src-tauri/tauri.conf.json"),
        "desktop/package.json": _json_version(ROOT / "desktop/package.json"),
        "desktop/package-lock.json": _json_version(ROOT / "desktop/package-lock.json"),
        "desktop/src-tauri/Cargo.toml": re.findall(r'^version = "([^"]+)"', cargo, flags=re.M)[:1],
        "desktop/src-tauri/Cargo.lock": re.findall(r'name = "enterprise-kb"\nversion = "([^"]+)"', lock.replace("\r\n", "\n")),
        "project/config.py": re.findall(r'app_version: str = "([^"]+)"', config),
    }


def changelog_versions() -> list[str]:
    """Released versions in CHANGELOG.md, newest first."""
    return re.findall(r"^## \[(\d+\.\d+\.\d+)\]", CHANGELOG.read_text(encoding="utf-8"), flags=re.M)


def _replace(path: Path, pattern: str, value: str, count: int = 1) -> None:
    raw = path.read_bytes().decode("utf-8")
    updated, made = re.subn(pattern, lambda m: m.group(1) + value + m.group(2), raw, count=count, flags=re.M)
    if not made:
        raise SystemExit(f"Version not found in {path}")
    path.write_bytes(updated.encode("utf-8"))


def set_version(version: str) -> None:
    _replace(ROOT / "desktop/src-tauri/tauri.conf.json", r'^(\s*"version": ")[^"]+(",)', version)
    _replace(ROOT / "desktop/package.json", r'^(\s*"version": ")[^"]+(",)', version)
    _replace(ROOT / "desktop/package-lock.json", r'^(\s*"version": ")[^"]+(",)', version, count=2)
    _replace(ROOT / "desktop/src-tauri/Cargo.toml", r'^(version = ")[^"]+(")', version)
    _replace(ROOT / "desktop/src-tauri/Cargo.lock", r'(name = "enterprise-kb"\r?\nversion = ")[^"]+(")', version)
    _replace(ROOT / "project/config.py", r'(app_version: str = ")[^"]+(")', version)


def open_changelog(version: str) -> bool:
    """Turn the 未发布 section into this version's section. False if it is already there."""
    if version in changelog_versions():
        return False
    raw = CHANGELOG.read_bytes().decode("utf-8")
    heading = "## [未发布]"
    if heading not in raw:
        raise SystemExit(f"No '{heading}' section in CHANGELOG.md")
    fresh = f"{heading}\n\n## [{version}] - {date.today().isoformat()}"
    CHANGELOG.write_bytes(raw.replace(heading, fresh, 1).encode("utf-8"))
    return True


def problems() -> list[str]:
    versions = read_versions()
    seen = {version for found in versions.values() for version in found}
    issues = [f"{name}: version not found" for name, found in versions.items() if not found]
    if len(seen) != 1:
        issues.append("files disagree: " + ", ".join(f"{name}={found}" for name, found in versions.items()))
    elif not changelog_versions() or changelog_versions()[0] != next(iter(seen)):
        issues.append(f"CHANGELOG.md has no section for {next(iter(seen))} at the top")
    return issues


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        raise SystemExit(__doc__)
    if argv:
        version = argv[0]
        if not VERSION.match(version):
            raise SystemExit(f"Not a version: {version} (expected 1.2.0)")
        set_version(version)
        print("CHANGELOG.md section opened" if open_changelog(version) else "CHANGELOG.md section already there")
    issues = problems()
    for issue in issues:
        print(issue, file=sys.stderr)
    if not issues:
        print(f"All files say {next(iter({v for found in read_versions().values() for v in found}))}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
