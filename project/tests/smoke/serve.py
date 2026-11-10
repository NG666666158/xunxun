"""Run the real backend against the fake model service, with a throwaway key store.

    python project/tests/smoke/serve.py --port 8110 --data-dir <folder>

No key is read from project/.env or from Windows Credential Manager, and a key typed into the
page is never written anywhere: the point is that tests cannot touch the real one.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import socket
import sys
import threading

PROJECT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--key", default="", help="start with this key already configured")
    args = parser.parse_args()

    models_port = free_port()
    os.environ.update(
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
        SILICONFLOW_API_KEY=args.key,
        LLM_API_KEY=args.key,
        SILICONFLOW_BASE_URL=f"http://127.0.0.1:{models_port}/v1",
        LLM_BASE_URL=f"http://127.0.0.1:{models_port}/v1",
        NO_PROXY="127.0.0.1,localhost",
        no_proxy="127.0.0.1,localhost",
    )
    sys.path[:0] = [str(PROJECT), str(Path(__file__).resolve().parent)]

    import uvicorn

    from fake_models import create_app

    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=models_port, log_level="warning"))
    threading.Thread(target=server.run, name="fake-models", daemon=True).start()

    import app as backend

    backend.main(["--host", "127.0.0.1", "--port", str(args.port), "--data-dir", args.data_dir])


if __name__ == "__main__":
    main()
