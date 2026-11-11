# PyInstaller spec for the desktop backend (FastAPI + LangGraph + embedded SQLite store).
#
#   .venv\Scripts\python.exe -m PyInstaller packaging\backend.spec --noconfirm ^
#       --distpath packaging\dist --workpath packaging\build
#
# Produces packaging/dist/rag-backend/ (onedir: rag-backend.exe + _internal/). onedir is
# deliberate: onefile unpacks to %TEMP% on every start, which is slower and is what
# antivirus heuristics flag most often. The Tauri shell ships this folder as a resource.

import json
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules, copy_metadata
from PyInstaller.utils.win32.versioninfo import (
    FixedFileInfo, StringFileInfo, StringStruct, StringTable, VarFileInfo, VarStruct, VSVersionInfo,
)

REPO = Path(SPECPATH).resolve().parent
PROJECT = REPO / "project"

# File properties for rag-backend.exe, written like the shell's and the installer's (product name
# and version from tauri.conf.json). Task Manager shows the description, and code signing through
# SignPath checks the product name and version of every file it signs.
TAURI = json.loads((REPO / "desktop" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
APP_VERSION = TAURI["version"]
_VERSION_NUMBERS = (*(int(part) for part in APP_VERSION.split(".")), 0)
VERSION_INFO = VSVersionInfo(
    ffi=FixedFileInfo(filevers=_VERSION_NUMBERS, prodvers=_VERSION_NUMBERS),
    kids=[
        StringFileInfo([StringTable("040904B0", [
            StringStruct("CompanyName", TAURI["bundle"]["publisher"]),
            StringStruct("FileDescription", f"{TAURI['productName']} 后台"),
            StringStruct("FileVersion", APP_VERSION),
            StringStruct("InternalName", "rag-backend"),
            StringStruct("OriginalFilename", "rag-backend.exe"),
            StringStruct("ProductName", TAURI["productName"]),
            StringStruct("ProductVersion", APP_VERSION),
        ])]),
        VarFileInfo([VarStruct("Translation", [0x0409, 1200])]),
    ],
)

# jieba ships POS tagging, keyword extraction and a Paddle LAC model we never use (~20 MB).
JIEBA_UNUSED = ("jieba/posseg", "jieba/analyse", "jieba/lac_small", "jieba\\posseg", "jieba\\analyse", "jieba\\lac_small")

a = Analysis(
    [str(PROJECT / "app.py")],
    pathex=[str(PROJECT)],
    # vec0.dll is loaded with sqlite3.load_extension, invisible to import analysis.
    binaries=collect_dynamic_libs("sqlite_vec"),
    # The UI is also served at /ui so the backend can be checked from a browser.
    # ocr_runtime.json pins the local-OCR wheels the app downloads on first use.
    datas=[(str(PROJECT / "ui" / "static"), "ui/static"), (str(PROJECT / "ingestion" / "ocr_runtime.json"), "ingestion")]
    + copy_metadata("keyring"),
    hiddenimports=[
        *collect_submodules("uvicorn"),
        "keyring.backends.Windows",
        "win32ctypes.core",
        # Word -> PDF previews drive the local Word through COM (library/preview.py, lazy imports).
        "pythoncom", "pywintypes", "win32com.client", "win32com.client.dynamic",
        "win32api", "win32con", "win32event", "win32gui", "win32process",
        # Our own packages: some are imported lazily (retrieval.factory, user settings).
        *collect_submodules("api"),
        *collect_submodules("core"),
        *collect_submodules("db"),
        *collect_submodules("ingestion"),
        *collect_submodules("library"),
        *collect_submodules("rag_agent"),
        *collect_submodules("services"),
        *collect_submodules("suggestions"),
        # The downloaded OCR wheels (OpenCV, Shapely) import numpy through the numpy.core
        # compatibility modules, which nothing in the app imports itself.
        *collect_submodules("numpy.core"),
        "retrieval.embedded_store",
        "retrieval.factory",
        "retrieval.hybrid_retriever",
        "retrieval.types",
    ],
    excludes=[
        # Optional Milvus backend and its stack (VECTOR_BACKEND=embedded in the desktop app).
        "pymilvus", "grpc", "grpcio", "retrieval.milvus_store",
        # Dev / demo / optional tooling present in the venv.
        "gradio", "gradio_client", "hf_gradio", "langfuse", "pandas", "matplotlib", "IPython",
        "tkinter", "pytest", "reportlab", "jieba.posseg", "jieba.analyse", "jieba.lac_small",
        "paddle",
        # pywin32's MFC GUI layer (6.5 MB); COM automation does not need it.
        "win32ui", "pywin.dialogs", "pywin.framework", "win32uiole", "dde",
        # The local-OCR stack is downloaded on first use (ingestion/ocr_runtime.py), not shipped.
        "rapidocr_onnxruntime", "onnxruntime", "cv2", "PIL", "shapely", "pyclipper", "sympy",
        "coloredlogs", "humanfriendly", "flatbuffers", "rapid_latex_ocr", "tokenizers", "chardet",
    ],
    noarchive=False,
)
a.datas = [entry for entry in a.datas if not entry[0].startswith(JIEBA_UNUSED)]
# The viewer libraries (PDF.js, SheetJS, docx-preview) are used by the desktop webview, which
# loads the UI from the Tauri bundle; the backend's /ui copy does not need them.
a.datas = [entry for entry in a.datas if not entry[0].replace("\\", "/").startswith("ui/static/vendor/")]
# Never ship a developer's project/.env (it may contain an API key).
a.datas = [entry for entry in a.datas if not entry[0].endswith(".env")]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="rag-backend",
    # Console subsystem so logs show when run by hand; the shell starts it with
    # CREATE_NO_WINDOW, so no console window appears in the desktop app.
    console=True,
    upx=False,
    icon=str(REPO / "desktop" / "src-tauri" / "icons" / "icon.ico"),
    version=VERSION_INFO,
)

coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="rag-backend")
