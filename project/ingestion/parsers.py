"""Parsers that normalize PDF, Word, Excel, Markdown, text and image files into sections."""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import pymupdf
from docx import Document as WordDocument
from openpyxl import load_workbook

from config import Settings, get_settings
from .chunker import CHUNKER_VERSION
from .figures import MODES as IMAGE_MODES, build_reader
from .images import IMAGE_EXTENSIONS, MODE_VERSION as IMAGE_MODE_VERSION, DEFAULT_IMAGE_MODE, parse_image
from .models import DocumentUnit, ParsedDocument
from .office import PPT_NEEDS_POWERPOINT, powerpoint_installed, ppt_to_pptx
from .pdf_layout import PDF_PARSER_VERSION, parse_pdf
from .slides import parse_pptx


# Version of the text each parser produces; unlisted types are version 1.
PARSER_VERSIONS: dict[str, int] = {".pdf": PDF_PARSER_VERSION}
# A PDF counts as scanned (and the UI offers to read it) when this share of its pages has no text.
SCANNED_PAGE_SHARE = 0.5
# Nearly all text: everything in them gets embedded, so their limit is lower (REPORT.md).
TEXT_LIKE = (".txt", ".md", ".xlsx")


def file_limit_mb(settings: Settings, suffix: str) -> int:
    """How big a file registered in place may be; browser uploads have their own, smaller limit."""
    if suffix.lower() in TEXT_LIKE:
        return settings.max_local_text_mb
    return settings.max_local_file_mb


def usable_extensions(extensions: tuple[str, ...]) -> tuple[str, ...]:
    """What this machine can read: a legacy .ppt needs PowerPoint."""
    if ".ppt" in extensions and not powerpoint_installed():
        return tuple(ext for ext in extensions if ext != ".ppt")
    return extensions


def pdf_image_mode(settings: Settings, image_mode: str | None = None) -> str:
    """How a PDF's figures and scanned pages are read: the file's own choice, else the setting."""
    if image_mode in IMAGE_MODES:
        return image_mode
    return getattr(settings, "image_understanding", "off")


def parser_version(path: str | Path, settings: Settings | None = None, image_mode: str | None = None) -> int:
    """Version of a file's index: parser * 1000 + chunker * 10 + image mode (PDFs, images).

    A file whose stored version differs is re-indexed when the library reconciles, so a
    new parser, new chunk boundaries or switching image understanding all reach files
    that are already indexed. (Older registries hold 1, 2 or parser * 100 + chunker.)
    ``image_mode`` is the file's own choice (an image file, or a scanned PDF the user had read).
    """
    suffix = Path(path).suffix.lower()
    version = PARSER_VERSIONS.get(suffix, 1) * 1000 + CHUNKER_VERSION * 10
    if suffix in IMAGE_EXTENSIONS:
        return version + IMAGE_MODE_VERSION.get(image_mode or DEFAULT_IMAGE_MODE, 1)
    if suffix == ".pdf":
        settings = settings or get_settings()
        mode = pdf_image_mode(settings, image_mode)
        version += IMAGE_MODES.index(mode) if mode in IMAGE_MODES else 0
        if mode == "local" and getattr(settings, "local_formula", False):
            version += 2  # 3: local OCR plus local formula recognition
    return version


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def stable_doc_id(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:24]


class MultiFormatParser:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._parsers: dict[str, Callable[[Path], list[DocumentUnit]]] = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".xlsx": self._parse_xlsx,
            ".md": self._parse_markdown,
            ".txt": self._parse_text,
            ".pptx": self._parse_pptx,
            ".ppt": self._parse_ppt,
        }

    def parse(
        self, file_path: str | Path, doc_id: str | None = None, image_mode: str | None = None, **pdf_options: Any
    ) -> ParsedDocument:
        """Parse a file; ``doc_id`` is the registry's file_id (the hash id is legacy).

        ``image_mode`` is the file's own choice: how an image file is read, or how a PDF's
        scanned pages are (overriding ``image_understanding``; "off" = the user declined).
        ``pdf_options`` go to ``parse_pdf`` (reading a long scan in passes, progress, pause);
        a deck gets the ``pause`` too.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in self._parsers and suffix not in IMAGE_EXTENSIONS:
            raise ValueError(f"不支持的文件格式：{suffix or '(无扩展名)'}")
        if not path.is_file():
            raise FileNotFoundError(path)
        limit = file_limit_mb(self.settings, suffix)
        if path.stat().st_size > limit * 1024 * 1024:
            raise ValueError(f"文件超过 {limit} MB 限制：{path.name}")

        version = parser_version(path, self.settings, image_mode)
        metadata: dict = {"parser_version": version}
        if suffix in IMAGE_EXTENSIONS:
            mode = image_mode or DEFAULT_IMAGE_MODE
            parsed_units, metadata = parse_image(path, mode, build_reader(self.settings, mode), self.settings)
            metadata["parser_version"] = version
        elif suffix == ".pdf":
            mode = pdf_image_mode(self.settings, image_mode)
            parsed_units, metadata = parse_pdf(path, build_reader(self.settings, mode), **pdf_options)
            parsed_units = self._flag_scanned(path, parsed_units, metadata, mode, image_mode)
            if metadata.get("pages_pending") and not any(unit.text.strip() for unit in parsed_units):
                # The first pages of a scan held no text yet: findable by name until more are read.
                parsed_units = [DocumentUnit(text=f"文件：{path.name}\n（扫描件，正在识别文字）", heading=path.stem)]
        elif suffix in (".pptx", ".ppt"):
            parsed_units, deck = self._parsers[suffix](path, pause=pdf_options.get("pause"))
            metadata.update(deck)
        else:
            parsed_units = self._parsers[suffix](path)
        units = [unit for unit in parsed_units if unit.text.strip()]
        if not units:
            raise ValueError(f"没有从 {path.name} 提取到可索引文本")
        return ParsedDocument(
            doc_id=doc_id or stable_doc_id(path),
            source=path.name,
            file_type=suffix,
            units=units,
            metadata=metadata,
        )

    @staticmethod
    def _flag_scanned(
        path: Path, units: list[DocumentUnit], metadata: dict, mode: str, image_mode: str | None
    ) -> list[DocumentUnit]:
        """A scanned PDF read without image understanding: index its name (so it can be found)
        and flag it, so the UI can ask whether to read its pages. Not asked again once declined."""
        scanned = metadata.get("image_only_pages") or []
        pages = int(metadata.get("pages") or 0)
        if mode != "off" or not scanned:
            return units
        if not any(unit.text.strip() for unit in units):
            units = [DocumentUnit(text=f"文件：{path.name}\n（扫描件，共 {pages} 页，尚未识别文字）", heading=path.stem)]
        if image_mode != "off" and len(scanned) >= max(1, SCANNED_PAGE_SHARE * pages):
            metadata["notice"] = "scanned_pdf"
        return units

    @staticmethod
    def _parse_pdf(path: Path) -> list[DocumentUnit]:
        return parse_pdf(path)[0]

    @staticmethod
    def _parse_docx(path: Path) -> list[DocumentUnit]:
        document = WordDocument(path)
        units: list[DocumentUnit] = []
        heading_stack: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            text = _clean_text("\n".join(buffer))
            if text:
                units.append(DocumentUnit(text=text, heading=" > ".join(heading_stack)))
            buffer.clear()

        for paragraph in document.paragraphs:
            text = _clean_text(paragraph.text)
            if not text:
                continue
            style_name = (paragraph.style.name or "").lower()
            match = re.search(r"heading\s*(\d+)|标题\s*(\d+)", style_name)
            if match:
                flush()
                level = int(match.group(1) or match.group(2) or 1)
                heading_stack[:] = heading_stack[: level - 1]
                heading_stack.append(text)
            else:
                buffer.append(text)
        flush()

        for table_index, table in enumerate(document.tables, start=1):
            rows = []
            for row in table.rows:
                values = [_clean_text(cell.text) for cell in row.cells]
                if any(values):
                    rows.append(" | ".join(values))
            if rows:
                units.append(
                    DocumentUnit(
                        text="\n".join(rows),
                        heading=f"表格 {table_index}",
                        locator=f"table:{table_index}",
                    )
                )
        return units

    @staticmethod
    def _parse_pptx(path: Path, pause: Callable[[], None] | None = None) -> tuple[list[DocumentUnit], dict]:
        return parse_pptx(str(path), pause)

    @staticmethod
    def _parse_ppt(path: Path, pause: Callable[[], None] | None = None) -> tuple[list[DocumentUnit], dict]:
        """A legacy .ppt has no readable structure of its own: PowerPoint saves it as .pptx first."""
        if not powerpoint_installed():
            raise ValueError(PPT_NEEDS_POWERPOINT)
        converted = ppt_to_pptx(str(path))
        try:
            return parse_pptx(converted, pause)
        finally:
            shutil.rmtree(Path(converted).parent, ignore_errors=True)

    @staticmethod
    def _parse_xlsx(path: Path) -> list[DocumentUnit]:
        workbook = load_workbook(path, read_only=True, data_only=True)
        units: list[DocumentUnit] = []
        try:
            for sheet in workbook.worksheets:
                # Rows are read as they come: a big sheet is never held in memory.
                rows = iter(sheet.iter_rows(values_only=True))
                first = next(rows, None)
                if first is None:
                    continue
                headers = [str(value).strip() if value is not None else f"列{index + 1}" for index, value in enumerate(first)]
                lines: list[str] = []
                start_row = 2
                for row_number, row in enumerate(rows, start=2):
                    pairs = [
                        f"{headers[index]}: {value}"
                        for index, value in enumerate(row)
                        if index < len(headers) and value not in (None, "")
                    ]
                    if pairs:
                        lines.append(f"第{row_number}行 | " + " | ".join(pairs))
                    if len(lines) == 30:
                        units.append(
                            DocumentUnit(
                                text="\n".join(lines),
                                heading=f"工作表：{sheet.title}",
                                locator=f"sheet:{sheet.title};rows:{start_row}-{row_number}",
                            )
                        )
                        lines = []
                        start_row = row_number + 1
                if lines:
                    end_row = start_row + len(lines) - 1
                    units.append(
                        DocumentUnit(
                            text="\n".join(lines),
                            heading=f"工作表：{sheet.title}",
                            locator=f"sheet:{sheet.title};rows:{start_row}-{end_row}",
                        )
                    )
        finally:
            workbook.close()
        return units

    @staticmethod
    def _parse_markdown(path: Path) -> list[DocumentUnit]:
        units: list[DocumentUnit] = []
        headings: list[str] = []
        buffer: list[str] = []

        def flush() -> None:
            text = _clean_text("\n".join(buffer))
            if text:
                units.append(DocumentUnit(text=text, heading=" > ".join(headings)))
            buffer.clear()

        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if match:
                flush()
                level = len(match.group(1))
                headings[:] = headings[: level - 1]
                headings.append(match.group(2).strip())
            else:
                buffer.append(line)
        flush()
        return units

    @staticmethod
    def _parse_text(path: Path) -> list[DocumentUnit]:
        text = _clean_text(path.read_text(encoding="utf-8-sig"))
        return [DocumentUnit(text=text, heading=path.stem)] if text else []
