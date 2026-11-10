"""The version is written in several files; the installer and updater depend on them agreeing."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("bump_version", ROOT / "scripts" / "bump_version.py")
bump_version = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bump_version)


def test_every_file_agrees_on_the_version_and_the_changelog_has_it():
    assert bump_version.problems() == []


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A copy of the files the script edits."""
    names = (
        "desktop/src-tauri/tauri.conf.json", "desktop/package.json", "desktop/package-lock.json",
        "desktop/src-tauri/Cargo.toml", "desktop/src-tauri/Cargo.lock", "project/config.py", "CHANGELOG.md",
    )
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    monkeypatch.setattr(bump_version, "ROOT", tmp_path)
    monkeypatch.setattr(bump_version, "CHANGELOG", tmp_path / "CHANGELOG.md")
    return tmp_path


def test_bumping_changes_every_file_and_opens_the_changelog_section(repo):
    assert bump_version.main(["9.8.7"]) == 0
    versions = bump_version.read_versions()
    assert all(found and set(found) == {"9.8.7"} for found in versions.values()), versions
    assert bump_version.changelog_versions()[0] == "9.8.7"
    changelog = (repo / "CHANGELOG.md").read_text(encoding="utf-8")
    assert changelog.index("## [未发布]") < changelog.index("## [9.8.7]")
    # Only the app's own lock entries moved.
    lock = json.loads((repo / "desktop/package-lock.json").read_text(encoding="utf-8"))
    assert lock["packages"]["node_modules/@tauri-apps/cli"]["version"] != "9.8.7"
    cargo_lock = (repo / "desktop/src-tauri/Cargo.lock").read_text(encoding="utf-8")
    assert cargo_lock.count('version = "9.8.7"') == 1


def test_a_disagreement_is_reported(repo):
    conf = repo / "desktop/package.json"
    conf.write_text(conf.read_text(encoding="utf-8").replace('"version": "', '"version": "0.', 1), encoding="utf-8")
    assert any("disagree" in issue for issue in bump_version.problems())


def test_a_missing_changelog_section_is_reported(repo):
    bump_version.set_version("9.9.9")
    assert any("CHANGELOG" in issue for issue in bump_version.problems())


def test_a_bad_version_is_refused(repo):
    with pytest.raises(SystemExit):
        bump_version.main(["1.2"])


def test_the_updater_stays_off_until_a_host_is_chosen():
    conf = json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    updater = conf["plugins"]["updater"]
    assert updater["endpoints"] == [] and updater["pubkey"] == ""
    assert updater["requireSignedVersion"] is True
    assert conf["bundle"]["createUpdaterArtifacts"] is False


def test_the_identity_that_keeps_conversations_and_data_is_unchanged():
    conf = json.loads((ROOT / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    # Conversations live in the webview's folder, named after the identifier: changing it
    # loses them. The installer's running-app check looks for this binary name.
    assert conf["identifier"] == "com.enterprisekb.desktop"
    assert conf["mainBinaryName"] == "EnterpriseKB"
