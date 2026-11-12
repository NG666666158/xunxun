"""Check a release's update files the way the app's updater will, before anyone downloads them.

    python packaging/verify_update.py --dir release --version 1.2.0 [--repo NG666666158/xunxun]

Needs the `cryptography` package (pip install cryptography). For Xunxun_<version>_x64-setup.exe,
its .sig and latest.json in --dir, it checks:

* the signature is by the key whose public half is in packaging/updater.conf.json (the one built
  into every installed copy); a different key means nobody can install the update;
* the signature covers these exact bytes: an installer changed after signing (for example
  Authenticode-signed afterwards) fails here, and in the app;
* the signed version is the release's version (`requireSignedVersion`);
* latest.json announces that version, carries the same signature, and points at the asset's URL.

The signature is minisign (Ed25519 over the BLAKE2b-512 hash of the file) as written by
`tauri build` or `tauri signer sign`.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPDATER_CONFIG = ROOT / "packaging" / "updater.conf.json"


def asset_name(version: str) -> str:
    return f"Xunxun_{version}_x64-setup.exe"


def _decode_lines(text: str) -> list[str]:
    return base64.b64decode(text.strip()).decode("utf-8").splitlines()


def verify_signature(data: bytes, signature: str, pubkey: str) -> str:
    """Return the signature's trusted comment; raise ValueError if the signature does not hold."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = base64.b64decode(_decode_lines(pubkey)[1])
    lines = _decode_lines(signature)
    if len(lines) < 4 or not lines[2].startswith("trusted comment: "):
        raise ValueError("not a minisign signature")
    sig = base64.b64decode(lines[1])
    if len(key) != 42 or key[:2] != b"Ed" or len(sig) != 74:
        raise ValueError("malformed key or signature")
    if sig[2:10] != key[2:10]:
        raise ValueError("signed with a different key than the one in the app (key ids differ)")
    public = Ed25519PublicKey.from_public_bytes(key[10:])
    message = hashlib.blake2b(data).digest() if sig[:2] == b"ED" else data
    trusted = lines[2][len("trusted comment: "):]
    try:
        public.verify(sig[10:], message)
    except InvalidSignature:
        raise ValueError("the signature does not match the file (was it changed after signing?)") from None
    try:
        public.verify(base64.b64decode(lines[3]), sig[10:] + trusted.encode("utf-8"))
    except InvalidSignature:
        raise ValueError("the trusted comment was changed after signing") from None
    return trusted


def signed_version(trusted_comment: str) -> str:
    return next((field[len("version:"):] for field in trusted_comment.split("\t") if field.startswith("version:")), "")


def problems(folder: Path, version: str, repo: str, pubkey: str) -> list[str]:
    setup = folder / asset_name(version)
    sig_file = Path(f"{setup}.sig")
    manifest_file = folder / "latest.json"
    missing = [path.name for path in (setup, sig_file, manifest_file) if not path.is_file()]
    if missing:
        return [f"missing in {folder}: {', '.join(missing)}"]
    signature = sig_file.read_text(encoding="utf-8").strip()
    issues = []
    try:
        trusted = verify_signature(setup.read_bytes(), signature, pubkey)
        if signed_version(trusted) != version:
            issues.append(f"the signature is for version {signed_version(trusted) or '(none)'}, not {version}")
    except ValueError as error:
        issues.append(f"{sig_file.name}: {error}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    platform = manifest.get("platforms", {}).get("windows-x86_64", {})
    url = f"https://github.com/{repo}/releases/download/v{version}/{asset_name(version)}"
    if manifest.get("version") != version:
        issues.append(f"latest.json announces {manifest.get('version')}, not {version}")
    if platform.get("signature", "").strip() != signature:
        issues.append("latest.json carries a different signature than the .sig file")
    if platform.get("url") != url:
        issues.append(f"latest.json points at {platform.get('url')}, expected {url}")
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repo", default="NG666666158/xunxun")
    parser.add_argument("--config", type=Path, default=UPDATER_CONFIG, help="the updater config with the public key")
    args = parser.parse_args(argv)
    pubkey = json.loads(args.config.read_text(encoding="utf-8"))["plugins"]["updater"]["pubkey"]
    issues = problems(args.dir, args.version, args.repo, pubkey)
    for issue in issues:
        print(f"::error::{issue}")
    if not issues:
        print(f"{asset_name(args.version)}: signature, signed version and latest.json are what the app expects")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
