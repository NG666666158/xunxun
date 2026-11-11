"""scripts/audit_deps.py: which pins pip-audit checks, and the ignore list's rules."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("audit_deps", ROOT / "scripts" / "audit_deps.py")
audit_deps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_deps)


def test_every_platforms_pins_are_checked_without_their_markers():
    text = (
        "# header\n"
        "aiosqlite==0.22.1\n"
        "cffi==2.1.1 ; platform_python_implementation != 'PyPy' and sys_platform == 'linux'\n"
        "pywin32==311 ; sys_platform == 'win32'\n"
        "\n"
    )
    assert audit_deps.pins(text) == ["aiosqlite==0.22.1", "cffi==2.1.1", "pywin32==311"]


def test_a_package_pinned_twice_goes_to_a_second_pass():
    groups = audit_deps.passes(["pycparser==3.11", "zipp==4.1.2", "PyCParser==3.12", "pillow==12.3.0"])
    assert groups == [["pycparser==3.11", "zipp==4.1.2", "pillow==12.3.0"], ["PyCParser==3.12"]]


def test_the_real_lock_splits_into_passes_without_duplicates():
    groups = audit_deps.passes(audit_deps.pins(audit_deps.LOCK.read_text(encoding="utf-8")))
    for group in groups:
        names = [audit_deps.canonical(pin) for pin in group]
        assert len(names) == len(set(names))
    assert sum(len(group) for group in groups) == len(audit_deps.pins(audit_deps.LOCK.read_text(encoding="utf-8")))
    assert all(";" not in pin for group in groups for pin in group)


def test_an_ignored_vulnerability_needs_a_reason():
    assert audit_deps.ignored("# comment\n\nPYSEC-2026-1845  pytest only runs in CI\n") == ["PYSEC-2026-1845"]
    with pytest.raises(SystemExit, match="PYSEC-2026-1845"):
        audit_deps.ignored("PYSEC-2026-1845\n")


def test_the_repository_ignore_list_is_valid():
    audit_deps.ignored(audit_deps.IGNORE.read_text(encoding="utf-8"))
