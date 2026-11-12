"""Uncaught exceptions reach backend.log (app.install_exception_hooks): the packaged backend
has no console, so before this a crashed thread, or the error that ended the process, left
no trace for the shell's "see backend.log" message to point at."""

from pathlib import Path
import os
import subprocess
import sys
import textwrap

from core.diagnostics import scrub_log

PROJECT = Path(__file__).resolve().parents[1]
SECRET = "sk-live-secret-0123456789"
DOCUMENT_TEXT = "合同第三条约定甲方应于本月底之前向乙方支付全部货款共计一百二十万元整"

SCRIPT = textwrap.dedent(
    f"""
    import sys, threading
    from pathlib import Path
    import app

    app.configure_logging(Path(sys.argv[1]), "INFO")
    app.install_exception_hooks(lambda: [{SECRET!r}, ""])

    def index_one_file():
        raise ValueError("reading failed: {DOCUMENT_TEXT} key={SECRET}")

    def ignored_exit():
        raise SystemExit(0)

    for target, name in ((index_one_file, "library-worker-1"), (ignored_exit, "quiet")):
        worker = threading.Thread(target=target, name=name)
        worker.start()
        worker.join()

    def serve():
        raise RuntimeError("Authorization: Bearer {SECRET}")

    serve()
    """
)


def _run(tmp_path: Path) -> tuple[subprocess.CompletedProcess, str]:
    logs = tmp_path / "logs"
    done = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(logs)], cwd=PROJECT, capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=120,
    )
    return done, (logs / "backend.log").read_text(encoding="utf-8")


def test_uncaught_exceptions_are_logged_with_traceback_and_without_secrets(tmp_path):
    done, log = _run(tmp_path)

    assert done.returncode == 1  # the main thread's error still ends the process
    assert "CRITICAL app.uncaught: Uncaught exception in thread library-worker-1" in log
    assert "CRITICAL app.uncaught: Uncaught exception in the main thread" in log
    assert log.count("Traceback (most recent call last):") == 2
    assert 'in index_one_file' in log and 'in serve' in log
    assert "ValueError: reading failed" in log and "RuntimeError" in log
    assert "quiet" not in log  # SystemExit in a thread is not a crash
    assert SECRET not in log and SECRET not in done.stderr
    assert "[已隐去]" in log


def test_exported_log_keeps_the_crash_but_not_its_message(tmp_path):
    _, log = _run(tmp_path)
    exported = "\n".join(scrub_log(log.splitlines(), [SECRET]))

    assert "Uncaught exception in thread library-worker-1" in exported
    assert "Traceback (most recent call last):" in exported
    assert "in index_one_file" in exported
    assert "\nValueError\n" in exported + "\n" and "RuntimeError" in exported
    assert DOCUMENT_TEXT not in exported and SECRET not in exported
