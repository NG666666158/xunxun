"""Reading figures and image-only pages of PDFs: a local OCR mode and a cloud vision mode.

The user picks one in Settings (``image_understanding``):

* ``off``   -- figures are skipped; their stray labels are dropped as noise (default).
* ``local`` -- nothing leaves the machine. A figure that has a text layer contributes
  its own labels; bitmap figures and image-only pages are read with RapidOCR.
* ``cloud`` -- figures are described by a vision model (``vision_model``) and
  image-only pages are transcribed by an OCR model (``vision_ocr_model``) on the
  configured SiliconFlow account. Better results, but page images are uploaded.
  Tables are re-read from their image by the OCR model too (complex headers, row
  labels spanning rows, tables set side by side), and replace the text-layer table
  when its numbers agree with the text layer.

Image files (.png, .jpg, ...) go through the same readers as one "image" region (see
``images.py``); their mode is chosen per file when they are added.

Results are cached on disk by image hash, so re-indexing a file does not call the
models again.
"""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import html
import logging
import os
from pathlib import Path
import queue
import re
import threading
from typing import Any, Callable, Iterator, Protocol
import unicodedata

from tenacity import retry

from services.model_clients import RETRY_BACKGROUND, background_calls
from . import ocr_runtime
from .priority import LOWEST, lower_current_thread, lower_new_threads

logger = logging.getLogger(__name__)

MODES = ("off", "local", "cloud")
# Part of the cache key and of the index version: bump when prompts or output format change.
FIGURE_FORMAT_VERSION = 1

DESCRIBE_PROMPT = (
    "这是一本书或文档中的一张图{caption}。请用中文写一段不超过 200 字的说明，供检索使用：\n"
    "1. 说明图展示的是什么（流程、结构、对比、数据、示例界面等）以及要点；\n"
    "2. 原样列出图中出现的关键文字标签、数字和公式（公式用 LaTeX）；\n"
    "3. 只描述图中能看到的内容，不要编造，不要复述图注以外的正文。\n"
    "图附近的正文：{context}"
)
OCR_PROMPT = "OCR:"
TABLE_PROMPT = "Table Recognition:"
IMAGE_PROMPT = (
    "这是用户资料库里的一张图片（文件名：{name}）。请用中文输出，供检索使用：\n"
    "1. 第一行用一句话说明这是什么图片（照片、截图、扫描件、图表、流程图、票据等）和主题；\n"
    "2. 接着原样转写图中所有能看清的文字，保持原来的顺序和换行；表格逐行写，单元格用 | 分隔；\n"
    "3. 只写图中能看到的内容，不要编造，不要解释。"
)
FORMULA_PROMPT = (
    "把图中的数学公式转写为 LaTeX（行内用 $…$），保留原有的中文说明文字。只输出转写结果，不要解释。"
    "图中没有公式时输出空字符串。"
)


@dataclass
class Figure:
    """A region to read: a figure (with caption and its text-layer labels) or a whole page."""

    page: int
    bbox: tuple[float, float, float, float]
    kind: str                          # "figure", "page", "formula", "table" or "image" (an image file)
    caption: str = ""
    context: str = ""
    labels: list[str] = field(default_factory=list)
    image: bytes = b""                 # PNG of the region (an image file may be JPEG)
    mime: str = "image/png"

    def key(self, mode: str, model: str) -> str:
        digest = hashlib.sha256()
        for part in (mode, model, str(FIGURE_FORMAT_VERSION), self.kind, self.caption):
            digest.update(part.encode("utf-8") + b"\0")
        digest.update(self.image)
        return digest.hexdigest()


class Reader(Protocol):
    mode: str
    model: str
    workers: int

    def read(self, figure: Figure) -> str: ...


class LocalReader:
    """Offline: text-layer labels when the figure has them, RapidOCR otherwise.

    With ``formulas`` (the optional formula component) display formulas are read with
    RapidLaTeXOCR; a result is kept only if it agrees with the PDF's own text of the
    formula, since this small model often misreads letters.
    """

    mode = "local"
    model = "rapidocr"
    formula_model = "rapid-latex-ocr"
    _formula_engine = None
    _lock = threading.Lock()
    # RapidOCR engines (separate ONNX sessions) shared by every file being read; created as
    # pages need them, up to layout()[0]. Measured in design/LIBRARY_SYNC_AND_IMAGES.md §6.
    _engines: "queue.Queue | None" = None
    _made = 0

    def __init__(self, data_dir: str | Path | None = None, formulas: bool = False):
        self.data_dir = data_dir
        self.reads_formulas = formulas
        self.workers = self.layout()[0]

    @staticmethod
    def layout() -> tuple[int, int]:
        """(engines, intra-op threads per engine): sessions of 2 threads on all but two logical
        processors (~30 MB each). Recognition is compute-bound: separate small sessions read
        1.4x more pages per second than one session shared by two threads, and 2 threads
        keep one page (one image) about twice as fast as 1 thread would. Their threads run
        at the lowest priority (see priority.py), so answers are not slowed."""
        usable = max(2, (os.cpu_count() or 4) - 2)
        return max(1, usable // 2), 2

    @contextmanager
    def engine(self) -> Iterator[Any]:
        """A free engine for one image, made on first need (up to the layout's count)."""
        with LocalReader._lock:
            if LocalReader._engines is None:
                # The desktop app downloads RapidOCR on first use (see ocr_runtime.py).
                if self.data_dir is not None and not ocr_runtime.activate(self.data_dir):
                    raise RuntimeError("本地识别组件尚未下载，请在设置 › 图片理解中下载")
                import cv2

                # OpenCV's own thread pool would run at normal priority; pages run side by side instead.
                cv2.setNumThreads(1)
                LocalReader._engines = queue.Queue()
            engines = LocalReader._engines
            count, threads = self.layout()
            if engines.empty() and LocalReader._made < count:
                from rapidocr_onnxruntime import RapidOCR

                engine, _ = lower_new_threads(lambda: RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1))
                engines.put(engine)
                LocalReader._made += 1
        engine = engines.get()
        try:
            yield engine
        finally:
            engines.put(engine)

    def formula_engine(self):
        with LocalReader._lock:
            if LocalReader._formula_engine is None:
                if self.data_dir is not None and not ocr_runtime.activate(self.data_dir, "formula"):
                    raise RuntimeError("本地公式识别组件尚未下载")
                from rapid_latex_ocr import LaTeXOCR

                files = ocr_runtime.model_files(self.data_dir, "formula") if self.data_dir is not None else {}
                paths = {
                    "image_resizer_path": files.get("image_resizer.onnx"),
                    "encoder_path": files.get("encoder.onnx"),
                    "decoder_path": files.get("decoder.onnx"),
                    "tokenizer_json": files.get("tokenizer.json"),
                }
                # Pinned, verified model files; the package would otherwise fetch unverified ones.
                LocalReader._formula_engine = LaTeXOCR(**{k: v for k, v in paths.items() if v is not None})
            return LocalReader._formula_engine

    def ocr_lines(self, image: bytes) -> list[str]:
        lower_current_thread(LOWEST)  # an answer being written goes first
        with self.engine() as engine:
            result, _ = engine(image)
        return [text.strip() for _, text, confidence in (result or []) if text.strip() and float(confidence) >= 0.6]

    def read_formula(self, figure: Figure) -> str:
        source = "".join(figure.labels)
        if not self.reads_formulas or _CJK.search(source):
            return ""  # the model only knows math; lines mixing Chinese come out as noise
        latex, _ = self.formula_engine()(figure.image)
        return f"${latex.strip()}$" if latex and latex_agrees(latex, source) else ""

    def read(self, figure: Figure) -> str:
        if figure.kind == "formula":
            return self.read_formula(figure)
        if figure.kind in ("page", "image"):
            return "\n".join(self.ocr_lines(figure.image))
        labels = figure.labels if len(figure.labels) >= 3 else self.ocr_lines(figure.image)
        unique = list(dict.fromkeys(label for label in labels if label and label not in figure.caption))
        return "、".join(unique)


_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATEX_COMMAND = re.compile(r"\\[a-zA-Z]+|[{}\s$~^_]")
# LaTeX names whose glyphs appear in the PDF text (\frac only draws a bar, so it just goes).
_SYMBOLS = {"times": "×", "cdot": "⋅", "approx": "≈", "mid": "∣", "sum": "∑", "sqrt": "√",
            "pi": "π", "alpha": "α", "gamma": "γ", "theta": "θ", "rho": "ρ", "mu": "μ",
            "ln": "ln", "log": "log", "exp": "exp", "max": "max", "min": "min"}


def latex_agrees(latex: str, text: str, threshold: float = 0.7) -> bool:
    """Does a transcription match the formula's text layer? Compares the visible characters."""
    from difflib import SequenceMatcher

    for name, symbol in _SYMBOLS.items():
        latex = re.sub(r"\\" + name + r"(?![a-zA-Z])", symbol, latex)
    plain = _LATEX_COMMAND.sub("", latex)
    reference = re.sub(r"\s", "", unicodedata.normalize("NFKC", text))
    if not plain or not reference:
        return False
    return SequenceMatcher(None, plain, reference, autojunk=False).ratio() >= threshold


class CloudReader:
    """SiliconFlow vision models: a description per figure, a transcription per page."""

    mode = "cloud"

    def __init__(self, settings):
        from openai import OpenAI

        self.client = OpenAI(api_key=settings.require_api_key, base_url=settings.siliconflow_base_url, timeout=180)
        self.model = settings.vision_model
        self.ocr_model = settings.vision_ocr_model
        self.workers = max(1, int(settings.vision_concurrency))

    def read(self, figure: Figure) -> str:
        if figure.kind == "page":
            model, prompt, limit = self.ocr_model, OCR_PROMPT, 2000
        elif figure.kind == "table":
            model, prompt, limit = self.ocr_model, TABLE_PROMPT, 3000
        elif figure.kind == "formula":
            model, prompt, limit = self.model, FORMULA_PROMPT, 400
        elif figure.kind == "image":
            model, prompt, limit = self.model, IMAGE_PROMPT.format(name=figure.context or "未命名"), 2000
        else:
            model, limit = self.model, 500
            caption = f"，图注：“{figure.caption}”" if figure.caption else ""
            prompt = DESCRIBE_PROMPT.format(caption=caption, context=figure.context[:300] or "（无）")
        image = f"data:{figure.mime};base64," + base64.b64encode(figure.image).decode()
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image}},
            {"type": "text", "text": prompt},
        ]}]
        return _clean(self._complete(model, messages, limit))

    @retry(**RETRY_BACKGROUND)
    def _complete(self, model: str, messages: list, limit: int) -> str:
        """One vision call: a few at once across all files, retried on dropped connections
        and 429/5xx (a refused image or a bad request fails at once)."""
        with background_calls.slot():
            response = self.client.chat.completions.create(
                model=model, messages=messages, max_tokens=limit, temperature=0.1,
            )
        return response.choices[0].message.content or ""


_OTSL = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")


def _cell_text(text: str) -> str:
    """Plain text of a recognised cell: inline LaTeX unwrapped, math letters folded."""
    text = re.sub(r"\\\(|\\\)|\$", "", text)
    text = re.sub(r"\\(?:text|mathbf|mathrm|mathit|textbf|operatorname)\s*\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[_^]\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"[{}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def otsl_rows(text: str) -> list[list[str]]:
    """Rows of cells from OTSL (PaddleOCR-VL's table format); merged cells repeat their value.

    ``<lcel>`` continues the cell to its left, ``<ucel>`` the one above, ``<xcel>`` both.
    The model sometimes replaces the very first ``<fcel>`` with a stray character: the text
    before the first tag counts as a cell only when that gives row one the usual width.
    """
    text = html.unescape(text)  # the tags sometimes come back as &lt;fcel&gt;
    rows: list[list[tuple[str, str]]] = [[]]
    head, *rest = _OTSL.split(text)
    for tag, content in zip(rest[0::2], rest[1::2]):
        if tag == "nl":
            rows.append([])
            continue
        rows[-1].append((tag, content))
    rows = [row for row in rows if row]
    if not rows:
        return []
    widths = sorted(len(row) for row in rows)
    head = "".join(char for char in head if not 0x1D400 <= ord(char) <= 0x1D7FF).strip()
    if head and len(rows[0]) + 1 == widths[len(widths) // 2]:
        rows[0].insert(0, ("fcel", head))
    grid: list[list[str]] = []
    for row in rows:
        cells: list[str] = []
        for column, (tag, content) in enumerate(row):
            if tag in ("lcel", "xcel") and cells:
                value = cells[-1]
            elif tag == "ucel" and grid and column < len(grid[-1]):
                value = grid[-1][column]
            else:
                value = _cell_text(content) if tag == "fcel" else ""
            cells.append(value)
        grid.append(cells)
    return grid


def table_text(grid: list[list[str]]) -> str:
    """One line per row, cells joined by " | " like the text-layer tables.

    Header rows (before the first row with a number) are folded into one line per
    column ("Top-20 NQ"), and a row spanning the whole table becomes a single cell.
    """
    numeric = re.compile(r"[-−–+±]?\d[\d.,/%]*")
    first = next((i for i, row in enumerate(grid) if any(numeric.fullmatch(c) for c in row[1:])), 0)
    lines = []
    if first > 1:
        spanning = [row for row in grid[:first] if len(set(row)) == 1]  # a section title across the table
        heads = [row for row in grid[:first] if len(set(row)) > 1]
        width = max((len(row) for row in heads), default=0)
        header = []
        for column in range(width):
            parts: list[str] = []
            for row in heads:
                value = row[column] if column < len(row) else ""
                if value and (not parts or parts[-1] != value):
                    parts.append(value)
            header.append(" ".join(parts))
        if header:
            lines.append(" | ".join(header))
        lines += [row[0] for row in spanning if row[0]]
        grid = grid[first:]
    for row in grid:
        distinct = [c for i, c in enumerate(row) if c and (i == 0 or c != row[i - 1])]
        if len(set(distinct)) == 1 and len(row) > 1:
            lines.append(distinct[0])
        elif any(row):
            lines.append(" | ".join(row))
    return "\n".join(lines)


def table_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:[.,]\d+)*", text)


def _clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class CachedReader:
    """Disk cache in front of a reader: ``<cache_dir>/<key[:2]>/<key>.txt``."""

    def __init__(self, reader: Reader, cache_dir: Path):
        self.reader = reader
        self.mode = reader.mode
        self.model = reader.model
        self.workers = reader.workers
        self.reads_formulas = getattr(reader, "reads_formulas", reader.mode == "cloud")
        self.cache_dir = cache_dir

    def read(self, figure: Figure) -> str:
        model = {
            "page": getattr(self.reader, "ocr_model", ""),
            "table": getattr(self.reader, "ocr_model", "") + "#otsl",  # raw OTSL, flattened by the parser
            "formula": getattr(self.reader, "formula_model", self.model),
        }
        key = figure.key(self.mode, model.get(figure.kind, self.model))
        # The data dir plus ~75 characters of cache path can pass Windows' 260-character limit.
        path = Path(ocr_runtime._long_path(self.cache_dir / key[:2] / f"{key}.txt"))
        if path.is_file():
            return path.read_text(encoding="utf-8")
        text = self.reader.read(figure)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text


def build_reader(settings, mode: str | None = None) -> Reader | None:
    """The reader for ``mode``; by default the PDF setting ``image_understanding``."""
    mode = mode or getattr(settings, "image_understanding", "off")
    if mode not in ("local", "cloud"):
        return None
    if mode == "local":
        reader: Reader = LocalReader(settings.data_dir, formulas=bool(getattr(settings, "local_formula", False)))
    else:
        reader = CloudReader(settings)
    return CachedReader(reader, Path(settings.data_dir) / "cache" / "figures")


def read_all(
    reader: Reader,
    figures: list[Figure],
    *,
    keep_empty: bool = False,
    on_done: Callable[[int], None] | None = None,
    pause: Callable[[], None] | None = None,
) -> dict[int, str]:
    """Read every figure (in parallel); returns index -> text (with ``keep_empty`` also the
    figures that were read but hold no text, as "").

    ``on_done(index)`` is called as each figure finishes; ``pause()`` before each one starts
    (the library waits there while a question is answered). A few failures are tolerated
    (the figure is simply left out). When most fail the service is likely down, so the
    error is raised and the file is retried later.
    """
    results: dict[int, str] = {}
    failures = 0

    def one(index: int) -> tuple[int, str | None]:
        if pause:
            pause()
        try:
            return index, reader.read(figures[index])
        except Exception as exc:  # network, content filter, OCR crash
            logger.warning("Reading figure on page %s failed: %s", figures[index].page, exc)
            return index, None
        finally:
            if on_done:
                on_done(index)

    with ThreadPoolExecutor(max_workers=max(1, reader.workers)) as pool:
        for index, text in pool.map(one, range(len(figures))):
            if text is None:
                failures += 1
            elif text.strip() or keep_empty:
                results[index] = text.strip()
    if figures and failures > max(2, len(figures) // 5):
        raise RuntimeError(f"图片识别失败 {failures}/{len(figures)} 张，稍后会自动重试")
    return results
