"""Pin the downloadable local-recognition components (project/ingestion/ocr_runtime.json).

The desktop installer leaves the heavy local-recognition stacks out; the app downloads
exactly these files -- verified by sha256 -- when the user first turns them on:

* ``ocr``     -- RapidOCR + onnxruntime + OpenCV + ... (text in figures, scanned pages)
* ``formula`` -- RapidLaTeXOCR, its tokenizer library and its ONNX models (formulas -> LaTeX)

Run this when changing versions or the Python of the packaged backend:

    .venv/Scripts/python.exe packaging/make_ocr_manifest.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import httpx

PYTHON_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
PLATFORM = "win_amd64"
# Wheels the packaged backend does not already contain (numpy, PyYAML, tqdm, requests,
# urllib3, idna, certifi, protobuf and packaging are bundled for other reasons).
COMPONENTS = {
    "ocr": {
        "label": "本地文字识别（RapidOCR）",
        "wheels": {
            "rapidocr-onnxruntime": "1.4.4",
            "onnxruntime": "1.23.2",
            "opencv-python-headless": "4.12.0.88",
            "pillow": "12.3.0",
            "pyclipper": "1.4.0",
            "shapely": "2.1.2",
            "six": "1.17.0",
            "coloredlogs": "15.0.1",
            "humanfriendly": "10.0",
            "pyreadline3": "3.5.6",
            "flatbuffers": "25.12.19",
        },
        "files": {},
    },
    "formula": {
        "label": "本地公式识别（RapidLaTeXOCR）",
        "wheels": {"rapid-latex-ocr": "0.0.9", "tokenizers": "0.23.3", "chardet": "7.7.0"},
        # The package downloads these itself on first use; we pin and verify them instead.
        "files": {
            name: f"https://github.com/RapidAI/RapidLaTeXOCR/releases/download/v0.0.0/{name}"
            for name in ("image_resizer.onnx", "encoder.onnx", "decoder.onnx", "tokenizer.json")
        },
    },
}
FILES_DIR = "rapid_latex_ocr_models"
OUT = Path(__file__).resolve().parents[1] / "project" / "ingestion" / "ocr_runtime.json"


def pick(files: list[dict]) -> dict:
    """The wheel for this Python on Windows x64, else an abi3 or pure wheel."""
    wheels = [f for f in files if f["filename"].endswith(".whl")]
    for test in (
        lambda n: f"-{PYTHON_TAG}-{PYTHON_TAG}-{PLATFORM}" in n,
        lambda n: "-abi3-" in n and n.endswith(f"{PLATFORM}.whl"),
        lambda n: n.endswith("-py3-none-any.whl") or n.endswith("-py2.py3-none-any.whl"),
    ):
        for wheel in wheels:
            if test(wheel["filename"]):
                return wheel
    raise SystemExit(f"no suitable wheel among {[w['filename'] for w in wheels]}")


def main() -> None:
    components = {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for key, spec in COMPONENTS.items():
            items = []
            for name, version in spec["wheels"].items():
                data = client.get(f"https://pypi.org/pypi/{name}/{version}/json").json()
                wheel = pick(data["urls"])
                items.append({"kind": "wheel", "name": name, "version": version, "filename": wheel["filename"],
                              "path": wheel["url"].split("files.pythonhosted.org", 1)[1],
                              "sha256": wheel["digests"]["sha256"], "size": wheel["size"]})
            for name, url in spec["files"].items():
                body = client.get(url).content
                items.append({"kind": "file", "filename": name, "urls": [url], "target": FILES_DIR,
                              "sha256": hashlib.sha256(body).hexdigest(), "size": len(body)})
            for item in items:
                print(f"{key:8} {item['filename']:66} {item['size'] / 1e6:6.1f} MB")
            components[key] = {"label": spec["label"], "items": items, "total_bytes": sum(i["size"] for i in items)}
    manifest = {
        "version": 2,
        "python": PYTHON_TAG,
        "platform": PLATFORM,
        # Wheels are tried on these hosts in order; mirrors keep PyPI's /packages/... layout.
        "hosts": ["https://files.pythonhosted.org", "https://pypi.tuna.tsinghua.edu.cn", "https://mirrors.aliyun.com/pypi"],
        "components": components,
    }
    OUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, component in components.items():
        print(f"{key}: {component['total_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
