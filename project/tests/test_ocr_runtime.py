"""Download-on-first-use of the local recognition components: mirrors, sha256 checks, atomic install."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import sys
import threading
import zipfile

import pytest

from ingestion import ocr_runtime
from ingestion.figures import Figure, LocalReader, latex_agrees


def _wheel(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()


MODEL = b"onnx model bytes"


@pytest.fixture
def mirrors():
    """Local "mirrors": /bad serves corrupted bytes, /good the real wheel; /models a model file."""
    good = _wheel({"fakeocr_pkg/__init__.py": "VALUE = 42\n"})
    served = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            served.append(self.path)
            if self.path.endswith("missing.whl"):
                self.send_response(404)
                self.end_headers()
                return
            body = b"corrupted" if self.path.startswith("/bad/") else MODEL if self.path.startswith("/models/") else good
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    yield base, good, served
    server.shutdown()


def _manifest(base: str, good: bytes, version: int = 990) -> dict:
    wheel = {"kind": "wheel", "name": "fakeocr", "version": "1", "filename": "fakeocr-1-py3-none-any.whl",
             "path": "/packages/fakeocr-1-py3-none-any.whl", "sha256": hashlib.sha256(good).hexdigest(), "size": len(good)}
    model = {"kind": "file", "filename": "encoder.onnx", "urls": [f"{base}/models/encoder.onnx"], "target": "models",
             "sha256": hashlib.sha256(MODEL).hexdigest(), "size": len(MODEL)}
    return {
        "version": version,
        "hosts": [f"{base}/bad", f"{base}/good"],
        "components": {"ocr": {"label": "测试", "items": [wheel, model], "total_bytes": len(good) + len(MODEL)}},
    }


def test_install_falls_back_to_mirror_verifies_sha256_and_activates(tmp_path, mirrors, monkeypatch):
    base, good, served = mirrors
    monkeypatch.setattr(ocr_runtime, "MANIFEST", _manifest(base, good))
    monkeypatch.setattr(ocr_runtime, "COMPONENTS", ("ocr",))
    monkeypatch.setitem(ocr_runtime.MODULES, "ocr", "fakeocr_pkg")
    installer = ocr_runtime.Installer(tmp_path, "ocr")
    assert installer.status()["downloaded"] is False

    installer.install()
    assert served == ["/bad/packages/fakeocr-1-py3-none-any.whl", "/good/packages/fakeocr-1-py3-none-any.whl",
                      "/models/encoder.onnx"]
    folder = ocr_runtime.runtime_dir(tmp_path, "ocr")
    assert (folder / "installed.json").is_file() and (folder / "fakeocr_pkg" / "__init__.py").is_file()
    assert ocr_runtime.model_files(tmp_path, "ocr") == {"encoder.onnx": folder / "models" / "encoder.onnx"}
    assert not any(p.name.startswith("staging-") for p in folder.parent.iterdir())

    try:
        assert ocr_runtime.activate(tmp_path, "ocr")
        import fakeocr_pkg

        assert fakeocr_pkg.VALUE == 42
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop("fakeocr_pkg", None)


def test_failed_download_leaves_nothing_behind_and_reports_error(tmp_path, mirrors, monkeypatch):
    base, good, _ = mirrors
    manifest = _manifest(base, good, version=991)
    manifest["components"]["ocr"]["items"][0]["path"] = "/packages/missing.whl"
    manifest["hosts"] = [f"{base}/good"]
    monkeypatch.setattr(ocr_runtime, "MANIFEST", manifest)
    installer = ocr_runtime.Installer(tmp_path, "ocr")
    installer._run()
    status = installer.status()
    assert status["state"] == "error" and "missing.whl" in status["error"]
    assert not ocr_runtime.runtime_dir(tmp_path, "ocr").exists()
    assert list(ocr_runtime.runtime_dir(tmp_path, "ocr").parent.iterdir()) == []


def test_pinned_manifest_matches_this_python():
    manifest = ocr_runtime.MANIFEST
    assert manifest["python"] == f"cp{sys.version_info.major}{sys.version_info.minor}"
    ocr = {item["name"] for item in manifest["components"]["ocr"]["items"] if item["kind"] == "wheel"}
    assert {"rapidocr-onnxruntime", "onnxruntime", "opencv-python-headless"} <= ocr
    formula = manifest["components"]["formula"]["items"]
    assert {i["filename"] for i in formula if i["kind"] == "file"} == {
        "image_resizer.onnx", "encoder.onnx", "decoder.onnx", "tokenizer.json"}
    for component in manifest["components"].values():
        for item in component["items"]:
            assert len(item["sha256"]) == 64
            assert item.get("path", "").startswith("/packages/") or item["urls"][0].startswith("https://")


# ---------------------------------------------------------------- local formula recognition


class FakeLatex:
    def __init__(self, answer):
        self.answer = answer
        self.calls = 0

    def __call__(self, image):
        self.calls += 1
        return self.answer, 0.1


def _formula(*labels):
    return Figure(page=1, bbox=(0, 0, 1, 1), kind="formula", labels=list(labels), image=b"png")


def test_local_formulas_are_kept_only_when_they_match_the_text_layer(monkeypatch):
    reader = LocalReader(formulas=True)
    good = FakeLatex(r"\mathrm{IDF}(t)=\ln{\frac{N}{\mathrm{DF}(t)}}")
    monkeypatch.setattr(LocalReader, "_formula_engine", good)
    assert reader.read(_formula("IDF(t) = ln", "N", "DF(t)")) == r"$\mathrm{IDF}(t)=\ln{\frac{N}{\mathrm{DF}(t)}}$"

    monkeypatch.setattr(LocalReader, "_formula_engine", FakeLatex(r"{\bf P}_{\mathrm{dSS}}(\Omega|_{\mathrm{K}}=1"))
    assert reader.read(_formula("Pass@k = 1 −(1 −p)k,")) == ""  # misread: dropped

    chinese = FakeLatex(r"\alpha")
    monkeypatch.setattr(LocalReader, "_formula_engine", chinese)
    assert reader.read(_formula("缓存输入单价为普通输入的𝛼倍")) == "" and chinese.calls == 0
    assert LocalReader(formulas=False).read(_formula("IDF(t) = ln")) == ""


def test_latex_agrees_compares_visible_characters():
    assert latex_agrees(r"\rho_{t}=\frac{\pi_{\theta}(a_{t}\mid s_{t})}{\mu(a_{t}\mid s_{t})}", "ρt= πθ(at∣st)μ(at∣st)")
    assert not latex_agrees(r"\mathcal{C}^{\ast}\big(s,\,\boldsymbol{a}\big)", "Q∗(s, a) = r+ γmaxa′ Q∗(s′, a′)")
