"""Local recognition components, downloaded on first use instead of shipped in the installer.

``ocr_runtime.json`` pins two components (see packaging/make_ocr_manifest.py):

* ``ocr``     -- RapidOCR + onnxruntime + OpenCV, ~77 MB: text in figures, scanned pages.
* ``formula`` -- RapidLaTeXOCR, its tokenizer and ONNX models, ~183 MB: formulas -> LaTeX.
  Optional; it reuses onnxruntime/OpenCV/Pillow from ``ocr``.

The installer downloads a component's wheels (PyPI, then two mirrors with the same
layout) and model files, checks every sha256, unpacks into a staging folder and only
then moves it to ``<data_dir>/ocr-runtime/v<version>/<component>/``. ``activate()`` puts
installed components on ``sys.path``. Running from source with the packages installed
needs no download.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from typing import Any
import zipfile

logger = logging.getLogger(__name__)

MANIFEST = json.loads((Path(__file__).with_name("ocr_runtime.json")).read_text(encoding="utf-8"))
COMPONENTS = tuple(MANIFEST["components"])
# What each component makes importable.
MODULES = {"ocr": "rapidocr_onnxruntime", "formula": "rapid_latex_ocr"}

# Model conversion/quantisation tools and test suites: never imported at runtime, and
# their deep paths can exceed Windows' 260-character limit.
SKIPPED = ("onnxruntime/tools/", "onnxruntime/transformers/", "onnxruntime/quantization/", "shapely/tests/")


def _long_path(path: Path) -> str:
    """``\\\\?\\`` prefix on Windows so unpacking is not limited to 260 characters."""
    text = str(path.resolve())
    return "\\\\?\\" + text if os.name == "nt" and not text.startswith("\\\\?\\") else text


def runtime_dir(data_dir: str | Path, component: str = "ocr") -> Path:
    return Path(data_dir) / "ocr-runtime" / f"v{MANIFEST['version']}" / component


def _installed(data_dir: str | Path, component: str) -> bool:
    return (runtime_dir(data_dir, component) / "installed.json").is_file()


def model_files(data_dir: str | Path, component: str) -> dict[str, Path]:
    """Downloaded model files of a component by name (empty when it is not installed)."""
    folder = runtime_dir(data_dir, component)
    return {
        item["filename"]: folder / item["target"] / item["filename"]
        for item in MANIFEST["components"][component]["items"]
        if item["kind"] == "file" and (folder / item["target"] / item["filename"]).is_file()
    }


def activate(data_dir: str | Path, component: str = "ocr") -> bool:
    """Make the component importable; False when it is neither installed in Python nor downloaded."""
    for name in COMPONENTS:  # formula needs ocr's onnxruntime/OpenCV, so add every installed one
        folder = runtime_dir(data_dir, name)
        if _installed(data_dir, name) and str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
            for dll_dir in (folder / "onnxruntime" / "capi", folder / "cv2"):
                if dll_dir.is_dir() and hasattr(os, "add_dll_directory"):
                    os.add_dll_directory(str(dll_dir))
            importlib.invalidate_caches()
    return importlib.util.find_spec(MODULES[component]) is not None


class Installer:
    """One background download per component; ``status()`` reports progress to the UI."""

    def __init__(self, data_dir: str | Path, component: str = "ocr"):
        self.data_dir = Path(data_dir)
        self.component = component
        self.spec = MANIFEST["components"][component]
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._state: dict[str, Any] = {"state": "idle", "done_bytes": 0, "error": ""}

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = dict(self._state)
        available = activate(self.data_dir, self.component) if state["state"] != "downloading" else False
        return {
            **state,
            "component": self.component,
            "label": self.spec["label"],
            "installed": available,
            "downloaded": _installed(self.data_dir, self.component),
            "total_bytes": self.spec["total_bytes"],
            "location": str(runtime_dir(self.data_dir, self.component)),
        }

    def start(self) -> dict[str, Any]:
        with self._lock:
            busy = self._thread is not None and self._thread.is_alive()
            if not busy and not _installed(self.data_dir, self.component):
                self._state = {"state": "downloading", "done_bytes": 0, "error": ""}
                self._thread = threading.Thread(target=self._run, name=f"{self.component}-download", daemon=True)
                self._thread.start()
        return self.status()

    def _progress(self, **fields: Any) -> None:
        with self._lock:
            self._state.update(fields)

    def _run(self) -> None:
        try:
            self.install()
            self._progress(state="done")
        except Exception as exc:
            logger.warning("Downloading %s failed: %s", self.component, exc)
            self._progress(state="error", error=str(exc)[:300] or type(exc).__name__)

    def install(self) -> None:
        import httpx

        target = runtime_dir(self.data_dir, self.component)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f"staging-{self.component}-", dir=target.parent))
        done = 0
        try:
            # trust_env: honours HTTP(S)_PROXY and, on Windows, the system proxy settings.
            with httpx.Client(timeout=httpx.Timeout(60, read=120), follow_redirects=True, trust_env=True) as client:
                for item in self.spec["items"]:
                    folder = staging / item.get("target", "")
                    folder.mkdir(parents=True, exist_ok=True)
                    path = self._download(client, item, folder, lambda n: self._progress(done_bytes=done + n))
                    if item["kind"] == "wheel":
                        with zipfile.ZipFile(path) as archive:
                            members = [m for m in archive.namelist() if not m.startswith(SKIPPED)]
                            archive.extractall(_long_path(staging), members)
                        path.unlink()
                    done += item["size"]
                    self._progress(done_bytes=done)
            (staging / "installed.json").write_text(json.dumps({"manifest": MANIFEST["version"]}), encoding="utf-8")
            if target.exists():
                shutil.rmtree(_long_path(target))
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(_long_path(staging), ignore_errors=True)

    @staticmethod
    def _download(client, item: dict, folder: Path, progress) -> Path:
        urls = item.get("urls") or [host + item["path"] for host in MANIFEST["hosts"]]
        errors = []
        for url in urls:
            destination = folder / item["filename"]
            digest = hashlib.sha256()
            received = 0
            try:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with destination.open("wb") as handle:
                        for block in response.iter_bytes(1 << 16):
                            handle.write(block)
                            digest.update(block)
                            received += len(block)
                            progress(received)
                if digest.hexdigest() != item["sha256"]:
                    raise ValueError("sha256 不匹配")
                return destination
            except Exception as exc:
                errors.append(f"{url.split('/')[2]}: {exc}")
                destination.unlink(missing_ok=True)
        raise RuntimeError(f"下载 {item['filename']} 失败：" + "；".join(errors))


_installers: dict[tuple[str, str], Installer] = {}
_installers_lock = threading.Lock()


def installer(data_dir: str | Path, component: str = "ocr") -> Installer:
    if component not in COMPONENTS:
        raise ValueError(f"unknown component: {component}")
    with _installers_lock:
        key = (str(Path(data_dir)), component)
        if key not in _installers:
            _installers[key] = Installer(data_dir, component)
        return _installers[key]
