"""The release workflow's own checks: tag vs version, the update signature, and draft-only releases."""

import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_check = _load("release_check", ROOT / "packaging" / "release_check.py")
verify_update = _load("verify_update", ROOT / "packaging" / "verify_update.py")


# --- tag, version and notes -------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """A copy of the files the version check reads."""
    names = (
        "desktop/src-tauri/tauri.conf.json", "desktop/package.json", "desktop/package-lock.json",
        "desktop/src-tauri/Cargo.toml", "desktop/src-tauri/Cargo.lock", "project/config.py", "CHANGELOG.md",
        "scripts/bump_version.py", "packaging/make_latest_json.py",
    )
    for name in names:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    return tmp_path


def _version(root: Path) -> str:
    return json.loads((root / "desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))["version"]


def test_the_matching_tag_passes_and_gives_the_changelog_section_as_notes(repo):
    version = _version(repo)
    found, notes, problems = release_check.check(f"v{version}", repo)
    assert problems == [] and found == version
    assert notes and not notes.startswith("## ")


def test_a_branch_build_checks_only_the_files(repo):
    assert release_check.check("", repo)[2] == []


def test_a_tag_for_another_version_fails_and_says_what_to_do(repo):
    problems = release_check.check("v99.0.0", repo)[2]
    assert len(problems) == 1 and "bump_version.py 99.0.0" in problems[0]


@pytest.mark.parametrize("tag", ["1.0.0", "v1.0", "v1.0.0-rc1", "release-1"])
def test_a_tag_not_shaped_like_v1_2_3_fails(repo, tag):
    assert any("not of the form" in problem for problem in release_check.check(tag, repo)[2])


def test_an_empty_changelog_section_fails(repo):
    version = _version(repo)
    changelog = repo / "CHANGELOG.md"
    text = changelog.read_text(encoding="utf-8").replace("\r\n", "\n")
    start = text.index(f"## [{version}]")
    end = text.index("\n## [", start + 1)
    changelog.write_text(text[:start] + f"## [{version}] - 2026-01-01\n" + text[end:], encoding="utf-8")
    assert any("empty" in problem for problem in release_check.check(f"v{version}", repo)[2])


def test_main_writes_the_notes_and_the_version_output(repo, tmp_path, monkeypatch):
    monkeypatch.setattr(release_check, "ROOT", repo)
    notes, output = tmp_path / "out" / "notes.md", tmp_path / "github_output"
    version = _version(repo)
    assert release_check.main(["--tag", f"v{version}", "--notes-out", str(notes), "--github-output", str(output)]) == 0
    assert output.read_text(encoding="utf-8") == f"version={version}\n"
    assert notes.read_text(encoding="utf-8").strip()


# --- the update signature -----------------------------------------------------------------


def test_the_shipped_public_key_is_a_minisign_ed25519_key():
    config = json.loads((ROOT / "packaging/updater.conf.json").read_text(encoding="utf-8"))
    lines = verify_update._decode_lines(config["plugins"]["updater"]["pubkey"])
    key = base64.b64decode(lines[1])
    assert len(key) == 42 and key[:2] == b"Ed"


class Signer:
    """Writes minisign signatures in the format `tauri signer sign` produces."""

    def __init__(self):
        ed25519 = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
        self.key = ed25519.Ed25519PrivateKey.generate()
        self.key_id = os.urandom(8)
        raw = self.key.public_key().public_bytes_raw()
        text = f"untrusted comment: minisign public key: test\n{base64.b64encode(b'Ed' + self.key_id + raw).decode()}\n"
        self.pubkey = base64.b64encode(text.encode()).decode()

    def sign(self, data: bytes, version: str, file: str = "setup.exe") -> str:
        signature = self.key.sign(hashlib.blake2b(data).digest())
        trusted = f"timestamp:1762800000\tfile:{file}\tversion:{version}"
        global_signature = self.key.sign(signature + trusted.encode())
        text = (
            "untrusted comment: signature from tauri secret key\n"
            f"{base64.b64encode(b'ED' + self.key_id + signature).decode()}\n"
            f"trusted comment: {trusted}\n{base64.b64encode(global_signature).decode()}\n"
        )
        return base64.b64encode(text.encode()).decode()


def _release(folder: Path, signer: Signer, version: str = "1.2.3", repo: str = "owner/xunxun", data: bytes = b"MZ installer") -> Path:
    name = verify_update.asset_name(version)
    (folder / name).write_bytes(data)
    signature = signer.sign(data, version)
    (folder / f"{name}.sig").write_text(signature, encoding="utf-8")
    manifest = {
        "version": version,
        "notes": "",
        "pub_date": "2026-11-11T00:00:00Z",
        "platforms": {"windows-x86_64": {
            "signature": signature,
            "url": f"https://github.com/{repo}/releases/download/v{version}/{name}",
        }},
    }
    (folder / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder / name


def test_a_correct_release_passes(tmp_path):
    signer = Signer()
    _release(tmp_path, signer)
    assert verify_update.problems(tmp_path, "1.2.3", "owner/xunxun", signer.pubkey) == []


def test_an_installer_changed_after_signing_fails(tmp_path):
    """What happens if the installer is Authenticode-signed and the .sig is not made again."""
    signer = Signer()
    setup = _release(tmp_path, signer)
    setup.write_bytes(setup.read_bytes() + b"authenticode signature")
    problems = verify_update.problems(tmp_path, "1.2.3", "owner/xunxun", signer.pubkey)
    assert len(problems) == 1 and "does not match the file" in problems[0]


def test_a_signature_by_another_key_fails(tmp_path):
    _release(tmp_path, Signer())
    problems = verify_update.problems(tmp_path, "1.2.3", "owner/xunxun", Signer().pubkey)
    assert len(problems) == 1 and "different key" in problems[0]


def test_a_signature_for_another_version_fails(tmp_path):
    signer = Signer()
    setup = _release(tmp_path, signer)
    signature = signer.sign(setup.read_bytes(), "1.2.2")
    Path(f"{setup}.sig").write_text(signature, encoding="utf-8")
    manifest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    manifest["platforms"]["windows-x86_64"]["signature"] = signature
    (tmp_path / "latest.json").write_text(json.dumps(manifest), encoding="utf-8")
    problems = verify_update.problems(tmp_path, "1.2.3", "owner/xunxun", signer.pubkey)
    assert problems == ["the signature is for version 1.2.2, not 1.2.3"]


def test_latest_json_must_agree_with_the_asset(tmp_path):
    signer = Signer()
    _release(tmp_path, signer, repo="someone/else")
    problems = verify_update.problems(tmp_path, "1.2.3", "owner/xunxun", signer.pubkey)
    assert len(problems) == 1 and "points at" in problems[0]


def test_missing_files_are_reported(tmp_path):
    assert "missing" in verify_update.problems(tmp_path, "1.2.3", "owner/xunxun", "")[0]


# --- the workflow ---------------------------------------------------------------------------


def _workflow() -> dict:
    return yaml.safe_load((ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8"))


def _commands(workflow: dict) -> str:
    return "\n".join(step.get("run", "") for job in workflow["jobs"].values() for step in job.get("steps", []))


def test_the_workflow_runs_on_version_tags_and_by_hand():
    triggers = _workflow()[True]  # YAML 1.1 reads the key `on` as True
    assert triggers["push"] == {"tags": ["v*"]} and "workflow_dispatch" in triggers


def test_the_workflow_only_ever_creates_drafts():
    commands = _commands(_workflow())
    creates = [line for line in commands.splitlines() if "gh release create" in line]
    assert creates and all("--draft" in line for line in creates)
    assert "draft=false" not in commands and "--draft false" not in commands
    assert "gh release delete" not in commands and "gh api" not in commands


def test_the_tests_and_the_version_check_come_before_the_build():
    jobs = _workflow()["jobs"]
    assert jobs["tests"]["uses"] == "./.github/workflows/ci.yml"
    assert {"version", "tests"} <= set(jobs["build"]["needs"])
    ci = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    assert "workflow_call" in ci[True]


def test_the_assets_keep_their_names():
    commands = _commands(_workflow())
    assert 'Xunxun_${env:VERSION}_x64-setup.exe' in commands
    assert "release/latest.json" in commands
