"""Check the pinned Python packages for known vulnerabilities with pip-audit, as CI does.

    python -m pip install pip-audit==2.10.1
    python scripts/audit_deps.py

requirements-lock.txt is resolved for every platform, so the environment markers are dropped
and every pin is checked, not only the ones this machine would install. pip-audit refuses a
package listed twice (one version per Python), so a second pin of a name goes to another pass.
Nothing is installed or resolved: the pins are looked up as they are.

To accept a known vulnerability for now, put its ID (or one of its aliases) in
.github/pip-audit-ignore.txt with the reason after it; a line without a reason fails the check.
Extra arguments go to pip-audit (e.g. --desc, -f json).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements-lock.txt"
IGNORE = ROOT / ".github" / "pip-audit-ignore.txt"
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def canonical(pin: str) -> str:
    match = NAME.match(pin)
    return re.sub(r"[-_.]+", "-", match.group(0)).lower() if match else pin


def pins(text: str) -> list[str]:
    """Each requirement without comments or its environment marker."""
    found = []
    for line in text.splitlines():
        requirement = line.split("#", 1)[0].split(";", 1)[0].strip()
        if requirement:
            found.append(requirement)
    return found


def passes(requirements: list[str]) -> list[list[str]]:
    """Groups in which no package appears twice, in file order."""
    groups: list[list[str]] = []
    for requirement in requirements:
        name = canonical(requirement)
        for group in groups:
            if all(canonical(other) != name for other in group):
                group.append(requirement)
                break
        else:
            groups.append([requirement])
    return groups


def ignored(text: str) -> list[str]:
    """The IDs to ignore: one per line, the reason after it."""
    ids = []
    for number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        vuln_id, *reason = line.split(None, 1)
        if not reason:
            raise SystemExit(f"{IGNORE.relative_to(ROOT)}:{number}: write why {vuln_id} is ignored after it")
        ids.append(vuln_id)
    return ids


def main(argv: list[str] | None = None) -> int:
    extra = sys.argv[1:] if argv is None else argv
    ignore = ignored(IGNORE.read_text(encoding="utf-8")) if IGNORE.is_file() else []
    groups = passes(pins(LOCK.read_text(encoding="utf-8")))
    worst = 0
    with tempfile.TemporaryDirectory() as folder:
        for index, group in enumerate(groups, 1):
            path = Path(folder) / f"pins-{index}.txt"
            path.write_text("\n".join(group) + "\n", encoding="utf-8")
            print(f"pass {index}/{len(groups)}: {len(group)} packages", flush=True)
            command = [sys.executable, "-m", "pip_audit", "-r", str(path), "--no-deps", "--disable-pip", "--strict"]
            command += ["--progress-spinner", "off"]
            for vuln_id in ignore:
                command += ["--ignore-vuln", vuln_id]
            # Descriptions hold characters a Chinese Windows console (GBK) cannot print.
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            worst = max(worst, subprocess.run(command + extra, check=False, env=env).returncode)
    return worst


if __name__ == "__main__":
    sys.exit(main())
