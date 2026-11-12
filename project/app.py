"""Run the FastAPI backend, which also serves the web UI at /ui.

Development: ``python project/app.py`` (0.0.0.0:8000, repository ``data/``, no token).
Desktop: the Tauri shell starts the packaged build with a random port, a per-launch token
(``RAG_APP_TOKEN`` env, or ``--token``) and ``--parent-pid`` so the backend exits with it.
"""

from __future__ import annotations

import argparse
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Callable, Iterable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enterprise knowledge base backend")
    parser.add_argument("--host", default=None, help="bind address (default 0.0.0.0, packaged: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token", default=None, help="require this token on every request (prefer RAG_APP_TOKEN)")
    parser.add_argument("--data-dir", default=None, help="where the database, parsed files and logs go")
    parser.add_argument("--parent-pid", type=int, default=None, help="exit when this process exits")
    return parser.parse_args(argv)


def apply_environment(args: argparse.Namespace, environ=os.environ) -> None:
    """Hand CLI choices to ``config.Settings`` before anything imports it."""
    if args.token:
        environ["RAG_APP_TOKEN"] = args.token
    if args.data_dir:
        environ["RAG_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())


def bind_host(args: argparse.Namespace, frozen: bool, token: str) -> str:
    if args.host:
        return args.host
    # The packaged app and any token-protected run stay on loopback.
    return "127.0.0.1" if frozen or token else "0.0.0.0"


def configure_logging(log_dir: Path, level: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(log_dir / "backend.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [file_handler]
    if sys.stderr is not None:
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        handlers.append(stream)
    logging.basicConfig(level=level.upper(), handlers=handlers, force=True)


def _known_secrets() -> list[str]:
    """The API key and the launch token, read when a crash happens (the key can change)."""
    try:
        from config import get_settings

        settings = get_settings()
        return [settings.siliconflow_api_key, settings.app_token]
    except Exception:
        return [os.environ.get("RAG_APP_TOKEN", "")]


def install_exception_hooks(secrets: Callable[[], Iterable[str]] = _known_secrets) -> None:
    """Uncaught exceptions, in the main thread or any other, go to backend.log with their
    traceback. The packaged backend has no console, so Python's own hooks printed them into
    nothing: a crashed worker thread, or the error that ended the process, left no trace.

    Known secrets are taken out before the text is written. The rest is scrubbed like any
    other log entry when the log is exported (core/diagnostics.py keeps the headline, the
    traceback positions and the exception's class, never its message)."""
    logger = logging.getLogger("app.uncaught")

    def write(where: str, exc_type, exc_value, exc_tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).rstrip()
        try:
            from core.diagnostics import scrub_text

            text = scrub_text(text, list(secrets()))
        except Exception:
            text = "\n".join(line for line in text.splitlines() if line.lstrip().startswith(("File ", "Traceback")))
        logger.critical("Uncaught exception in %s\n%s", where, text)

    previous_hook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def main_hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous_hook(exc_type, exc_value, exc_tb)
            return
        try:
            write("the main thread", exc_type, exc_value, exc_tb)
        except Exception:
            previous_hook(exc_type, exc_value, exc_tb)

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        name = args.thread.name if args.thread is not None else "an unknown thread"
        try:
            write(f"thread {name}", args.exc_type, args.exc_value, args.exc_traceback)
        except Exception:
            previous_thread_hook(args)

    sys.excepthook = main_hook
    threading.excepthook = thread_hook


def watch_parent(pid: int, interval: float = 1.0) -> None:
    """Exit when the desktop shell is gone, even if it crashed without stopping us."""

    def alive() -> bool:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
            if not handle:
                return False
            try:
                return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    def loop() -> None:
        while alive():
            time.sleep(interval)
        logging.getLogger(__name__).info("Parent process %s exited; shutting down", pid)
        os._exit(0)

    threading.Thread(target=loop, name="parent-watch", daemon=True).start()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    apply_environment(args)

    import uvicorn

    from config import get_settings, is_frozen

    settings = get_settings()
    configure_logging(settings.log_dir, settings.log_level)
    install_exception_hooks()
    if args.parent_pid:
        watch_parent(args.parent_pid)

    from api.main import app

    host = bind_host(args, is_frozen(), settings.app_token)
    logging.getLogger(__name__).info(
        "Backend on %s:%s, data dir %s, token %s",
        host,
        args.port,
        settings.data_dir,
        "required" if settings.app_token else "off",
    )
    uvicorn.run(app, host=host, port=args.port, reload=False, log_config=None, access_log=False)


if __name__ == "__main__":
    main()
