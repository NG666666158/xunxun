"""Image files (.png, .jpg, ...): read with the figure readers in the mode chosen per file.

* ``local`` -- RapidOCR on this machine (the default; nothing is uploaded).
* ``cloud`` -- the vision model describes the image and transcribes its text. The image,
  scaled to ``image_max_side`` pixels on its long side, is uploaded to SiliconFlow.
* ``name``  -- only the file name, folder and date are indexed; no recognition at all.

Every mode indexes the name, folder and date, so an image stays findable when no text
could be read (a photo without text, or local OCR not downloaded yet). Readings are
cached by image hash (``figures.CachedReader``), so a moved or re-saved image is not
read again.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

from . import ocr_runtime
from .figures import Figure, Reader
from .models import DocumentUnit

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
IMAGE_MODES = ("local", "cloud", "name")
DEFAULT_IMAGE_MODE = "local"
# Index version of an image per mode (see parsers.parser_version).
MODE_VERSION = {"local": 1, "cloud": 2, "name": 3}
MAX_PAGES = 20  # multi-page TIFF scans
_DATE_TAGS = (36867, 36868, 306)  # EXIF DateTimeOriginal, DateTimeDigitized, DateTime

# For the import estimate, measured in design/LIBRARY_SYNC_AND_IMAGES.md §4 (six test images:
# screenshots, a receipt, a chart, a book page, a photo; RapidOCR on this laptop's CPU,
# Qwen3-VL-32B on SiliconFlow): seconds for one image alone (median), and seconds per image
# when many run side by side as the library runs them. "name" is two embedding requests.
SECONDS_PER_IMAGE = {"local": 3.4, "cloud": 25.0, "name": 0.6}
SECONDS_PER_IMAGE_IN_BATCH = {"local": 2.5, "cloud": 12.9, "name": 0.15}
# Image jobs the library runs side by side, per mode ("cloud" uses vision_concurrency).
PARALLEL_JOBS = {"local": 2, "name": 4}


def estimate_seconds(mode: str, count: int) -> float:
    """Seconds to read ``count`` images in ``mode`` (not less than one image takes)."""
    if count <= 0:
        return 0.0
    return max(SECONDS_PER_IMAGE[mode], count * SECONDS_PER_IMAGE_IN_BATCH[mode])


def is_image(path: str | Path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def parallel_jobs(mode: str, settings: Any) -> int:
    if mode == "cloud":
        return max(1, int(getattr(settings, "vision_concurrency", 6)))
    return PARALLEL_JOBS.get(mode, 1)


def _pil():
    try:
        from PIL import Image

        return Image
    except ImportError:
        return None


def taken_at(path: Path) -> datetime:
    """When the photo was taken (EXIF) if Pillow can tell, else the file's modification time."""
    image_module = _pil()
    if image_module is not None:
        try:
            with image_module.open(path) as image:
                exif = image.getexif()
                values = {**dict(exif.get_ifd(0x8769)), **dict(exif)}
                for tag in _DATE_TAGS:
                    value = str(values.get(tag) or "").strip()
                    if value:
                        return datetime.strptime(value[:19], "%Y:%m:%d %H:%M:%S")
        except Exception:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def load_pages(path: Path, max_side: int = 2000) -> list[tuple[bytes, str]]:
    """Each page of an image file as ``(bytes, mime)``, at most ``max_side`` px on the long side.

    PyMuPDF (always bundled) decodes PNG, JPEG, BMP and TIFF and paints transparency on
    white; WebP needs Pillow, which comes with the local OCR component. JPEG originals
    stay JPEG (smaller uploads), everything else becomes PNG.
    """
    import pymupdf

    as_jpeg = path.suffix.lower() in (".jpg", ".jpeg")
    try:
        native = pymupdf.Pixmap(str(path))
        document = pymupdf.open(str(path))
    except Exception:
        return _load_with_pil(path, max_side)
    pages: list[tuple[bytes, str]] = []
    with document:
        first = document[0].rect
        pixels_per_point = native.width / max(first.width, 1)
        for page in list(document)[:MAX_PAGES]:
            width, height = page.rect.width * pixels_per_point, page.rect.height * pixels_per_point
            scale = min(1.0, max_side / max(width, height, 1))
            zoom = pixels_per_point * scale
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            if as_jpeg:
                pages.append((pixmap.tobytes("jpg", jpg_quality=88), "image/jpeg"))
            else:
                pages.append((pixmap.tobytes("png"), "image/png"))
    return pages


def _load_with_pil(path: Path, max_side: int) -> list[tuple[bytes, str]]:
    import io

    image_module = _pil()
    if image_module is None:
        # No decoder (WebP before the OCR component is downloaded): the vision model takes it as is.
        return [(path.read_bytes(), "image/webp" if path.suffix.lower() == ".webp" else "application/octet-stream")]
    pages = []
    with image_module.open(path) as image:
        for index in range(min(getattr(image, "n_frames", 1), MAX_PAGES)):
            image.seek(index)
            frame = image.convert("RGBA")
            background = image_module.new("RGBA", frame.size, (255, 255, 255, 255))
            frame = image_module.alpha_composite(background, frame).convert("RGB")
            frame.thumbnail((max_side, max_side))
            buffer = io.BytesIO()
            frame.save(buffer, format="PNG")
            pages.append((buffer.getvalue(), "image/png"))
    return pages


def forget_readings(path: str | Path, settings: Any) -> int:
    """Delete the cached readings of an image file (local OCR and the cloud model, at the
    current models), e.g. once the user keeps it by name only. Returns how many were removed."""
    from .figures import LocalReader

    path = Path(path)
    if not path.is_file():
        return 0
    cache = Path(settings.data_dir) / "cache" / "figures"
    if not cache.is_dir():
        return 0
    models = {("local", LocalReader.model), ("cloud", str(getattr(settings, "vision_model", "") or ""))}
    removed = 0
    for number, (image, mime) in enumerate(load_pages(path, int(getattr(settings, "image_max_side", 2000))), start=1):
        figure = Figure(page=number, bbox=(0, 0, 0, 0), kind="image", context=path.name, image=image, mime=mime)
        for mode, model in models:
            key = figure.key(mode, model)
            target = Path(ocr_runtime._long_path(cache / key[:2] / f"{key}.txt"))
            if target.is_file():
                target.unlink()
                removed += 1
    return removed


def header(path: Path) -> str:
    """The lines every image's text starts with: what it is, where it is, when it is from."""
    return f"图片：{path.name}\n文件夹：{path.parent.name or path.anchor}\n日期：{taken_at(path):%Y-%m-%d}"


def parse_image(
    path: Path, mode: str, reader: Reader | None, settings: Any
) -> tuple[list[DocumentUnit], dict[str, Any]]:
    """Units and metadata for one image file read in ``mode``.

    A failed reading raises (network, rate limit), so the file is retried later instead
    of being indexed by name only. Local OCR that is not downloaded yet gives the name
    only, with ``notice = "ocr_unavailable"``: the file is read once the component arrives.
    """
    mode = mode if mode in IMAGE_MODES else DEFAULT_IMAGE_MODE
    # Also puts the component's Pillow on sys.path (EXIF dates, WebP) when it is downloaded.
    ocr_available = ocr_runtime.activate(settings.data_dir)
    top = header(path)
    metadata: dict[str, Any] = {"image_mode": mode, "pages": 1}
    if mode == "local" and not ocr_available:
        metadata["notice"] = "ocr_unavailable"
        return [DocumentUnit(text=top, heading=path.stem)], metadata
    if mode == "name" or reader is None:
        return [DocumentUnit(text=top, heading=path.stem)], metadata
    pages = load_pages(path, int(getattr(settings, "image_max_side", 2000)))
    metadata["pages"] = len(pages)
    label = "〔图片内容〕" if mode == "cloud" else "〔图中文字〕"
    units: list[DocumentUnit] = []
    read_chars = 0
    for number, (image, mime) in enumerate(pages, start=1):
        text = reader.read(Figure(page=number, bbox=(0, 0, 0, 0), kind="image", context=path.name, image=image, mime=mime))
        text = (text or "").strip()
        read_chars += len(text)
        if len(pages) == 1:
            units.append(DocumentUnit(text=f"{top}\n{label}\n{text}" if text else top, heading=path.stem))
        elif text:
            units.append(DocumentUnit(text=f"{top}\n{label}（第 {number} 页）\n{text}", heading=path.stem,
                                      locator=f"page:{number}"))
    if not units:
        units.append(DocumentUnit(text=top, heading=path.stem))
    metadata["text_chars"] = read_chars
    return units, metadata
