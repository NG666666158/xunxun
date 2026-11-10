"""Install an old build, upgrade it, and check a copy of real data survives. Leaves no trace.

    python packaging/test_upgrade.py --old <old-setup.exe> --new <new-setup.exe> --data <data copy> --work <folder> [--rename]

Why this is safe beside an installed, running app:

* Everything is installed into --work with /D=. The installer only closes an app running from the
  folder it installs to (Restart Manager on that path), so the real one keeps running.
* /NS keeps it from writing shortcuts. The uninstaller only removes shortcuts pointing at its own folder.
* The install registration in HKCU is exported first, and restored and compared at the end.
* The backend runs on a copy of --data, on a port of its own, and its model service is unreachable.
* The built backend is also tried on an empty folder against the fake model service, for what only
  a packaged build can get wrong: modules import analysis missed (decks, tables, saved state).

It cannot start the desktop shell: a second instance would just focus the real app.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SMOKE = Path(__file__).resolve().parents[1] / "project" / "tests" / "smoke"
OLD_NAME = "企业知识库"
NEW_NAME = "寻寻"
UNINSTALL = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall"
KEYS = {
    "location": r"HKCU\Software\Enterprise KB",
    # Where the renamed build keeps its install location (publisher 寻寻).
    "location_new": r"HKCU\Software\寻寻",
    "uninstall_old": rf"{UNINSTALL}\{OLD_NAME}",
    "uninstall_new": rf"{UNINSTALL}\{NEW_NAME}",
}
TOKEN = "upgrade-test-token"
# The version the installer under test should carry: the one this checkout says.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bump_version import read_versions  # noqa: E402

NEW_VERSION = read_versions()["project/config.py"][0]


class Failed(RuntimeError):
    pass


def check(condition: bool, what: str, results: list[dict], detail: object = "") -> None:
    results.append({"ok": bool(condition), "check": what, "detail": detail})
    print(("  ok   " if condition else "  FAIL ") + what + (f"  ({detail})" if detail != "" else ""), flush=True)


def reg_export(key: str, path: Path) -> bool:
    return subprocess.run(["reg", "export", key, str(path), "/y"], capture_output=True).returncode == 0


def reg_delete(key: str) -> None:
    subprocess.run(["reg", "delete", key, "/f"], capture_output=True)


def reg_exists(key: str) -> bool:
    return subprocess.run(["reg", "query", key], capture_output=True).returncode == 0


def save_registry(folder: Path) -> dict[str, Path | None]:
    folder.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path | None] = {}
    for name, key in KEYS.items():
        path = folder / f"{name}.reg"
        saved[name] = path if reg_export(key, path) else None
    return saved


def restore_registry(saved: dict[str, Path | None], folder: Path) -> list[str]:
    """Back to exactly what it was; returns the keys that differ afterwards."""
    for name, key in KEYS.items():
        reg_delete(key)
        if saved[name]:
            subprocess.run(["reg", "import", str(saved[name])], capture_output=True, check=True)
    differ = []
    for name, key in KEYS.items():
        after = folder / f"{name}.after.reg"
        exists = reg_export(key, after)
        if exists != bool(saved[name]) or (exists and after.read_bytes() != saved[name].read_bytes()):
            differ.append(key)
    return differ


def run_installer(setup: Path, target: Path, *flags: str) -> int:
    # /D= must come last and takes no quotes.
    command = f'"{setup}" /S /NS {" ".join(flags)} /D={target}'.replace("  ", " ")
    return subprocess.run(command, shell=True, timeout=600).returncode


def file_list(folder: Path) -> set[str]:
    return {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}


class Backend:
    def __init__(self, install: Path, data: Path, log: Path, models: str = "http://127.0.0.1:9/v1"):
        self.install, self.data, self.log, self.models = install, data, log, models
        self.process: subprocess.Popen | None = None
        self.base = ""

    def __enter__(self) -> "Backend":
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base = f"http://127.0.0.1:{port}"
        env = {
            **os.environ, "RAG_APP_TOKEN": TOKEN, "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1",
            # Nothing leaves this machine from a test, whatever key is stored.
            "SILICONFLOW_BASE_URL": self.models, "LLM_BASE_URL": self.models,
            "SILICONFLOW_API_KEY": "sk-good-upgrade-test", "LLM_API_KEY": "sk-good-upgrade-test",
        }
        handle = self.log.open("ab")
        self.process = subprocess.Popen(
            [str(self.install / "backend" / "rag-backend.exe"), "--port", str(port), "--data-dir", str(self.data)],
            stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=self.install / "backend",
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if self.get("/health/live")[0] == 200:
                    return self
            except (OSError, urllib.error.URLError):
                pass
            if self.process.poll() is not None:
                raise Failed(f"backend exited with {self.process.returncode}; see {self.log}")
            time.sleep(0.2)
        raise Failed("backend did not start")

    def __exit__(self, *_exc) -> None:
        self.process.terminate()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()

    def call(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        request = urllib.request.Request(
            f"{self.base}/api/v1{path}", method=method, data=None if body is None else json.dumps(body).encode("utf-8"),
            headers={"X-App-Token": TOKEN, "Content-Type": "application/json"},
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=120) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, {}

    def get(self, path: str) -> tuple[int, dict]:
        return self.call("GET", path)

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self.call("POST", path, body)

    def wait_library(self, settle: float = 90) -> dict:
        """Idle, or done with its first pass: nothing queued and the read count steady for
        ``settle`` seconds. With the model service unreachable, files that need it fail and are
        retried with growing delays for many minutes, so the library is seldom idle for long."""
        deadline = time.time() + 600
        steady_since, last = time.time(), None
        while time.time() < deadline:
            code, status = self.get("/library/status")
            if code == 200:
                if not status["busy"]:
                    return status
                indexed = status["counts"].get("indexed")
                if status.get("queued") or indexed != last:
                    steady_since, last = time.time(), indexed
                elif time.time() - steady_since >= settle:
                    return status
            time.sleep(0.5)
        raise Failed("library stayed busy")


def counts(db: Path) -> dict:
    conn = sqlite3.connect(db)
    try:
        return {
            "files": conn.execute("SELECT count(*) FROM files WHERE deleted_at IS NULL").fetchone()[0],
            "chunks": conn.execute("SELECT count(*) FROM chunks").fetchone()[0],
            "roots": conn.execute("SELECT count(*) FROM watch_roots WHERE deleted_at IS NULL").fetchone()[0],
            "indexed": conn.execute("SELECT count(*) FROM files WHERE status='indexed' AND deleted_at IS NULL").fetchone()[0],
        }
    finally:
        conn.close()


def nothing_lost(before: dict, after: dict) -> bool:
    """Same files, chunks and folders. More may count as read: a newer version knows more kinds of
    file, while one that newly can be read fails here for want of a model, and that is expected."""
    same = all(after[key] == before[key] for key in ("files", "chunks", "roots"))
    return same and after["indexed"] >= before["indexed"]


def check_conversations(install: Path, data: Path, log: Path, results: list[dict], before: dict) -> None:
    """Conversations move in from the old place, get backed up, and a restore keeps newer ones."""
    old = {"id": "from-localstorage", "title": "旧对话", "updated": 1.0, "messages": [{"role": "user", "text": "旧问题"}]}
    with Backend(install, data, log) as app:
        check(app.get("/settings")[0] == 200 and app.get("/conversations")[0] == 200, "conversations answer", results)
        code, moved = app.post("/conversations/import", {"conversations": [old]})
        check(code == 200 and moved.get("imported") == 1, "old conversations move into the data folder", results)
        check((data / "conversations.sqlite3").exists(), "conversations.sqlite3 is in the data folder", results)
        code, made = app.post("/data-safety/backup", {})
        backup = (made.get("backups") or [{}])[0]
        check(code == 200 and "conversations.sqlite3" in backup.get("files", []), "a backup covers them", results, backup.get("files"))
        code, _ = app.call("PUT", "/conversations/newer", {"title": "备份之后", "updated": 2.0, "messages": []})
        check(code == 200, "a newer conversation is saved", results)
        code, _ = app.post("/data-safety/restore", {"name": backup.get("name", "")})
        check(code == 200, "restore is accepted", results)
    with Backend(install, data, log) as app:
        ids = {item["id"] for item in app.get("/conversations")[1].get("conversations", [])}
        check({"from-localstorage", "newer"} <= ids, "restore does not take back newer conversations", results, sorted(ids))
        check(app.get("/data-safety")[1].get("state") == "ok", "checks pass after the restore", results)
        app.wait_library()
    after = counts(data / "knowledge.sqlite3")
    check(nothing_lost(before, after), "the index is intact after the restore", results, after)


def check_features(install: Path, work: Path, log: Path, results: list[dict]) -> None:
    """The built backend on an empty folder and fake models: only the plumbing, not answer quality."""
    sys.path.insert(0, str(SMOKE))
    import uvicorn
    from fake_models import create_app
    from office_files import write_deck, write_workbook

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    models = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=models.run, daemon=True).start()
    data, docs = work / "data-features", work / "docs-features"
    docs.mkdir()
    deck = write_deck(docs / "路演材料.pptx", ["市场规模约三百亿", "团队有十二人"])
    book = write_workbook(docs / "报名名单.xlsx", [["姓名", "部门"], ["甲", "人事"], ["乙", "财务"], ["丙", "人事"]])
    base = f"http://127.0.0.1:{port}/v1"
    try:
        with Backend(install, data, log, models=base) as app:
            code, _ = app.post("/library/import", {"paths": [str(deck), str(book)]})
            check(code == 200, "a deck and a workbook are added", results)
            deadline = time.time() + 120
            while time.time() < deadline and app.get("/library/status")[1].get("counts", {}).get("indexed", 0) < 2:
                time.sleep(0.5)
            status = app.get("/library/status")[1]
            check(status.get("counts", {}).get("indexed") == 2, "both are read", results, status.get("counts"))
            code, answer = app.post("/chat", {"question": "总结一下《路演材料》", "thread_id": "summary"})
            check(code == 200 and "已通读全部 2 张幻灯片" in answer.get("answer", ""), "a whole-file summary", results, answer.get("answer", "")[-40:])
            code, answer = app.post("/chat", {"question": "《报名名单》一共多少条", "thread_id": "table"})
            check(code == 200 and "共 3 条" in answer.get("answer", ""), "a table question", results, answer.get("answer", "")[:40])
            files = {row["name"]: row["file_id"] for row in app.get("/files?limit=5")[1].get("files", [])}
            code, slides = app.get(f"/documents/{files.get(deck.name, 'none')}/slides")
            check(code == 200 and len(slides.get("slides", [])) == 2, "a deck can be shown without PowerPoint", results)
            code, notes = app.get("/data-safety")
            check(code == 200 and notes.get("state") == "ok", "checks pass with conversations in use", results)
        # The state of those turns is on disk for the next run.
        conn = sqlite3.connect(data / "conversations.sqlite3")
        try:
            saved = {row[0] for row in conn.execute("SELECT DISTINCT thread_id FROM checkpoints")}
        finally:
            conn.close()
        check(saved == {"summary", "table"}, "agent state is saved", results, sorted(saved))
        with Backend(install, data, log, models=base) as app:
            code, answer = app.post("/chat", {"question": "总结一下《路演材料》", "thread_id": "summary"})
            check(code == 200 and "已通读全部 2 张幻灯片" in answer.get("answer", ""), "the same conversation carries on after a restart", results)
    finally:
        models.should_exit = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="a copy of the data folder; copied again before use")
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--old-version", default="1.0.0", help="the version --old installs")
    parser.add_argument("--rename", action="store_true", help="--new is the build renamed to 寻寻")
    parser.add_argument("--as-update", action="store_true", help="install the way the updater does (/UPDATE)")
    args = parser.parse_args()

    results: list[dict] = []
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    old_dir, new_dir, fresh_dir = work / "app-old", work / ("app-new" if args.rename else "app-old"), work / "app-fresh"
    data = work / "data"
    shutil.copytree(args.data, "\\\\?\\" + str(data), ignore=shutil.ignore_patterns("backups", "recovery-pending.json", "logs"))
    (data / "preferences.json").write_text(
        json.dumps({k: v for k, v in json.loads((data / "preferences.json").read_text(encoding="utf-8")).items()
                    if k not in ("last_run_version", "onboarding_done", "data_notice_seen")}),
        encoding="utf-8",
    )
    saved = save_registry(work / "registry")
    print("registry saved:", {k: bool(v) for k, v in saved.items()}, flush=True)
    real_app = Path(os.environ["LOCALAPPDATA"]) / OLD_NAME / "EnterpriseKB.exe"
    real_mtime = real_app.stat().st_mtime if real_app.exists() else None
    log = work / "backend.log"
    # Only a rename installed as an update writes these; they are removed again below.
    shortcuts = [
        Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs" / f"{NEW_NAME}.lnk",
        Path(os.environ["USERPROFILE"]) / "Desktop" / f"{NEW_NAME}.lnk",
    ]
    try:
        # ---------------- the old version ----------------
        print("install old", flush=True)
        check(run_installer(args.old, old_dir) == 0 and (old_dir / "EnterpriseKB.exe").exists(), "old version installs", results)
        with Backend(old_dir, data, log) as old:
            version = old.get("/health/live")[1].get("version")
            check(version == args.old_version, "old backend runs on the data copy", results, version)
            status = old.wait_library()
            previous_backups = {b["name"] for b in old.get("/data-safety")[1].get("backups", [])}
        before = counts(data / "knowledge.sqlite3")
        print("before", before, status["counts"], flush=True)

        # ---------------- the new version over it ----------------
        print("install new", flush=True)
        flags = ("/UPDATE",) if args.as_update else ()
        check(run_installer(args.new, new_dir, *flags) == 0 and (new_dir / "EnterpriseKB.exe").exists(), "new version installs over it", results)
        if args.rename:
            check(not (old_dir / "EnterpriseKB.exe").exists(), "old install folder is gone", results)
            check(not old_dir.exists(), "nothing is left in the old folder", results, sorted(file_list(old_dir))[:5] if old_dir.exists() else "")
            check(not reg_exists(KEYS["uninstall_old"]), "old uninstall entry is gone", results)
            check(reg_exists(KEYS["uninstall_new"]), "new uninstall entry exists", results)
            if args.as_update:
                for shortcut in shortcuts:
                    check(shortcut.exists(), f"shortcut is back: {shortcut.parent.name}", results)
        with Backend(new_dir, data, log) as new:
            version = new.get("/health/live")[1].get("version")
            check(version == NEW_VERSION, "new backend starts on the same data", results, version)
            code, settings = new.get("/settings")
            check(code == 200, "settings answer", results)
            safety = settings.get("data_safety", {})
            check(safety.get("state") == "ok", "database check passes", results, f"{safety.get('check_ms')} ms")
            # An old version with data safety (1.1.0) kept its own copy on its first run: look for
            # the one this upgrade made, labelled with the version it replaced.
            upgrade = [b for b in safety.get("backups", []) if b.get("reason") == "upgrade"
                       and b.get("app_version") == args.old_version and b.get("name") not in previous_backups]
            check(len(upgrade) == 1, "a copy was kept before the upgrade", results, upgrade[0].get("size") if upgrade else "")
            check(settings.get("onboarding_done") is False and settings.get("data_notice_seen") is False, "the notice is still to be shown", results)
            status = new.wait_library()
            check(new.get("/documents")[0] == 200, "documents list", results)
            check(new.get("/files?limit=5")[0] == 200, "file list", results)
        after = counts(data / "knowledge.sqlite3")
        print("after", after, status["counts"], flush=True)
        check(nothing_lost(before, after), "files, chunks and folders are unchanged", results, after)
        check((data / "backups").is_dir(), "backups live in the data folder", results)
        check_conversations(new_dir, data, log, results, before)

        # ---------------- leftovers against a clean install ----------------
        check(run_installer(args.new, fresh_dir) == 0, "clean install for comparison", results)
        extra = sorted(file_list(new_dir) - file_list(fresh_dir))
        check(not [name for name in extra if name.endswith((".exe", ".py"))], "no stale programs", results, f"{len(extra)} stale files: {extra[:5]}")
        missing = sorted(file_list(fresh_dir) - file_list(new_dir))
        check(not missing, "nothing missing compared with a clean install", results, missing[:5])

        # ---------------- what only a packaged build can get wrong ----------------
        check_features(fresh_dir, work, log, results)

        # ---------------- uninstalling ----------------
        code = subprocess.run(f'"{new_dir / "uninstall.exe"}" /S _?={new_dir}', shell=True).returncode
        check(code == 0 and not (new_dir / "EnterpriseKB.exe").exists(), "uninstall removes the app", results)
        check((data / "knowledge.sqlite3").exists(), "uninstall leaves the data", results)
    except Failed as exc:
        check(False, str(exc), results)
    finally:
        if args.rename and args.as_update:
            for shortcut in shortcuts:
                shortcut.unlink(missing_ok=True)
        differ = restore_registry(saved, work / "registry")
        check(not differ, "install registration restored exactly", results, differ or "")
        if real_mtime is not None:
            check(real_app.stat().st_mtime == real_mtime, "the real install was not touched", results)
    failed = [r for r in results if not r["ok"]]
    (work / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(results) - len(failed)}/{len(results)} checks passed; results in {work / 'results.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
