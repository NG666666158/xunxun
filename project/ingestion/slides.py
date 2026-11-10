"""PowerPoint decks (.pptx): text, tables, charts and notes per slide, read from the file's XML.

A deck is a zip of XML parts plus its media. Only the XML parts are read, so a deck full of
pictures or video costs no memory beyond its text (python-pptx would load all of it).
Reading order is the slides' geometry, top to bottom and left to right: the order shapes
are stored in is their z-order. Text inside pictures is not read.

``read_deck`` also gives each shape's position and style, which is what the reader's
rendering without PowerPoint draws (``/documents/{id}/slides``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import posixpath
import re
from typing import Any, Callable
import zipfile

from lxml import etree

from .models import DocumentUnit

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
C = "http://schemas.openxmlformats.org/drawingml/2006/chart"
DGM = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
NS = {"a": A, "p": P, "r": R, "c": C, "dgm": DGM}

# A part bigger than this is not a slide's XML; refuse instead of filling memory.
MAX_PART_BYTES = 64 * 1024 * 1024
TITLE_PLACEHOLDERS = {"title", "ctrTitle"}
# Filled in by PowerPoint on every slide; they say nothing about the slide.
FURNITURE_PLACEHOLDERS = {"dt", "ftr", "sldNum", "hdr"}
# Top coordinates within this share of the slide's height count as one row of shapes.
ROW_TOLERANCE = 0.02
BROWSER_IMAGES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp")

_parser = etree.XMLParser(resolve_entities=False, no_network=True, huge_tree=False, recover=False)


@dataclass
class Run:
    text: str
    size: float = 0.0  # points; 0 = not set on the run
    bold: bool = False
    color: str = ""


@dataclass
class Paragraph:
    runs: list[Run] = field(default_factory=list)
    align: str = ""

    @property
    def text(self) -> str:
        return "".join(run.text for run in self.runs)


@dataclass
class Shape:
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    kind: str = "text"  # text | table | chart | diagram | picture
    title: bool = False
    paragraphs: list[Paragraph] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)  # charts and diagrams, as plain lines
    image: str = ""  # picture: the part's name inside the zip

    @property
    def text(self) -> str:
        if self.kind == "table":
            return "\n".join(" | ".join(cell for cell in row) for row in self.rows if any(row))
        if self.kind in ("chart", "diagram"):
            return "\n".join(self.lines)
        return "\n".join(p.text for p in self.paragraphs if p.text.strip())


@dataclass
class Slide:
    number: int
    hidden: bool = False
    shapes: list[Shape] = field(default_factory=list)
    notes: str = ""

    @property
    def pictures(self) -> int:
        return sum(1 for shape in self.shapes if shape.kind == "picture")

    @property
    def body(self) -> str:
        return "\n".join(text for text in (shape.text.strip() for shape in self.shapes) if text)

    @property
    def title(self) -> str:
        return _slide_title(self)


@dataclass
class Deck:
    width: int
    height: int
    slides: list[Slide]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ").replace("\x0b", "\n").replace("\r\n", "\n")
    text = re.sub(r"[ \t　]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class _Package:
    """The XML parts of a deck, each parsed at most once."""

    def __init__(self, archive: zipfile.ZipFile):
        self.archive = archive
        self.names = set(archive.namelist())
        self._rels: dict[str, dict[str, tuple[str, str]]] = {}

    def xml(self, name: str) -> etree._Element | None:
        if name not in self.names:
            return None
        if self.archive.getinfo(name).file_size > MAX_PART_BYTES:
            raise ValueError(f"{name} 超过 {MAX_PART_BYTES // (1024 * 1024)} MB")
        return etree.fromstring(self.archive.read(name), _parser)

    def rels(self, part: str) -> dict[str, tuple[str, str]]:
        """rId -> (relationship type's last word, target part name) of one part."""
        if part not in self._rels:
            folder, base = posixpath.split(part)
            root = self.xml(posixpath.join(folder, "_rels", f"{base}.rels"))
            found: dict[str, tuple[str, str]] = {}
            for rel in (root if root is not None else ()):
                if rel.get("TargetMode") == "External":
                    continue
                target = rel.get("Target") or ""
                name = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(folder, target))
                found[rel.get("Id") or ""] = ((rel.get("Type") or "").rsplit("/", 1)[-1], name)
            self._rels[part] = found
        return self._rels[part]

    def related(self, part: str, kind: str) -> str:
        return next((name for rel, name in self.rels(part).values() if rel == kind), "")


def _paragraphs(body: etree._Element) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    for node in body.findall("a:p", NS):
        props = node.find("a:pPr", NS)
        paragraph = Paragraph(align=(props.get("algn") or "") if props is not None else "")
        for item in node:
            tag = etree.QName(item).localname
            if tag == "br":
                paragraph.runs.append(Run("\n"))
            elif tag == "fld" and (item.get("type") or "").startswith("slidenum"):
                continue
            elif tag in ("r", "fld"):
                text = item.findtext("a:t", default="", namespaces=NS)
                if text:
                    paragraph.runs.append(_run(text, item.find("a:rPr", NS)))
        paragraphs.append(paragraph)
    return paragraphs


def _run(text: str, props: etree._Element | None) -> Run:
    if props is None:
        return Run(text)
    color = props.find("a:solidFill/a:srgbClr", NS)
    return Run(
        text,
        size=int(props.get("sz") or 0) / 100,
        bold=props.get("b") in ("1", "true"),
        color=(color.get("val") or "") if color is not None else "",
    )


@dataclass(frozen=True)
class _Transform:
    """From a group's child coordinates to the slide's."""

    sx: float = 1.0
    sy: float = 1.0
    dx: float = 0.0
    dy: float = 0.0

    def box(self, x: int, y: int, w: int, h: int) -> tuple[int, int, int, int]:
        return (
            round(x * self.sx + self.dx), round(y * self.sy + self.dy),
            round(w * self.sx), round(h * self.sy),
        )


def _xfrm_box(node: etree._Element | None) -> tuple[int, int, int, int] | None:
    if node is None:
        return None
    off, ext = node.find("a:off", NS), node.find("a:ext", NS)
    if off is None or ext is None:
        return None
    return int(off.get("x") or 0), int(off.get("y") or 0), int(ext.get("cx") or 0), int(ext.get("cy") or 0)


def _group_transform(group: etree._Element, parent: _Transform) -> _Transform:
    xfrm = group.find("p:grpSpPr/a:xfrm", NS)
    box = _xfrm_box(xfrm)
    child_off = xfrm.find("a:chOff", NS) if xfrm is not None else None
    child_ext = xfrm.find("a:chExt", NS) if xfrm is not None else None
    if box is None or child_off is None or child_ext is None:
        return parent
    cx, cy = int(child_ext.get("cx") or 0), int(child_ext.get("cy") or 0)
    sx = box[2] / cx if cx else 1.0
    sy = box[3] / cy if cy else 1.0
    return _Transform(
        parent.sx * sx,
        parent.sy * sy,
        parent.dx + parent.sx * (box[0] - int(child_off.get("x") or 0) * sx),
        parent.dy + parent.sy * (box[1] - int(child_off.get("y") or 0) * sy),
    )


class _SlideReader:
    def __init__(self, package: _Package, part: str, number: int):
        self.package = package
        self.part = part
        self.number = number
        self._layout: dict[tuple[str, str], tuple[int, int, int, int]] | None = None

    def read(self) -> Slide:
        root = self.package.xml(self.part)
        slide = Slide(self.number, hidden=root.get("show") == "0" if root is not None else False)
        tree = root.find("p:cSld/p:spTree", NS) if root is not None else None
        if tree is not None:
            self._walk(tree, _Transform(), slide.shapes)
        slide.notes = self._notes()
        return slide

    def _walk(self, tree: etree._Element, transform: _Transform, shapes: list[Shape]) -> None:
        for node in tree:
            tag = etree.QName(node).localname
            if tag == "grpSp":
                self._walk(node, _group_transform(node, transform), shapes)
            elif tag == "sp":
                self._text_shape(node, transform, shapes)
            elif tag == "graphicFrame":
                self._graphic_frame(node, transform, shapes)
            elif tag == "pic":
                self._picture(node, transform, shapes)

    def _text_shape(self, node: etree._Element, transform: _Transform, shapes: list[Shape]) -> None:
        placeholder = node.find("p:nvSpPr/p:nvPr/p:ph", NS)
        kind = placeholder.get("type") or "body" if placeholder is not None else ""
        if kind in FURNITURE_PLACEHOLDERS:
            return
        body = node.find("p:txBody", NS)
        if body is None:
            return
        paragraphs = _paragraphs(body)
        if not any(p.text.strip() for p in paragraphs):
            return
        box = _xfrm_box(node.find("p:spPr/a:xfrm", NS))
        if box is None and placeholder is not None:
            box = self._inherited_box(placeholder)
        x, y, w, h = transform.box(*(box or (0, 0, 0, 0)))
        shapes.append(Shape(x, y, w, h, title=kind in TITLE_PLACEHOLDERS, paragraphs=paragraphs))

    def _inherited_box(self, placeholder: etree._Element) -> tuple[int, int, int, int] | None:
        """A placeholder without its own position takes the one on the slide's layout."""
        if self._layout is None:
            self._layout = {}
            layout = self.package.related(self.part, "slideLayout")
            root = self.package.xml(layout) if layout else None
            for shape in (root.iterfind("p:cSld/p:spTree/p:sp", NS) if root is not None else ()):
                ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
                box = _xfrm_box(shape.find("p:spPr/a:xfrm", NS))
                if ph is not None and box is not None:
                    self._layout[(ph.get("type") or "body", ph.get("idx") or "")] = box
                    self._layout.setdefault((ph.get("type") or "body", ""), box)
        kind = placeholder.get("type") or "body"
        return self._layout.get((kind, placeholder.get("idx") or "")) or self._layout.get((kind, ""))

    def _graphic_frame(self, node: etree._Element, transform: _Transform, shapes: list[Shape]) -> None:
        x, y, w, h = transform.box(*(_xfrm_box(node.find("p:xfrm", NS)) or (0, 0, 0, 0)))
        data = node.find("a:graphic/a:graphicData", NS)
        if data is None:
            return
        table = data.find("a:tbl", NS)
        if table is not None:
            rows = [
                [_clean("\n".join(t.text or "" for t in cell.iterfind(".//a:t", NS))) for cell in row.findall("a:tc", NS)]
                for row in table.findall("a:tr", NS)
            ]
            if any(any(row) for row in rows):
                shapes.append(Shape(x, y, w, h, kind="table", rows=rows))
            return
        chart = data.find("c:chart", NS)
        if chart is not None:
            lines = self._chart_lines(chart.get(f"{{{R}}}id") or "")
            if lines:
                shapes.append(Shape(x, y, w, h, kind="chart", lines=lines))
            return
        diagram = data.find("dgm:relIds", NS)
        if diagram is not None:
            lines = self._diagram_lines(diagram.get(f"{{{R}}}dm") or "")
            if lines:
                shapes.append(Shape(x, y, w, h, kind="diagram", lines=lines))

    def _chart_lines(self, rel_id: str) -> list[str]:
        target = self.package.rels(self.part).get(rel_id)
        root = self.package.xml(target[1]) if target else None
        if root is None:
            return []
        lines: list[str] = []
        title = "".join(t.text or "" for t in root.iterfind("c:chart/c:title//a:t", NS)).strip()
        if title:
            lines.append(f"图表：{title}")
        for series in root.iterfind(".//c:ser", NS):
            name = "".join(v.text or "" for v in series.iterfind("c:tx//c:v", NS)).strip()
            categories = [v.text or "" for v in series.iterfind("c:cat//c:pt/c:v", NS)]
            values = [v.text or "" for v in series.iterfind("c:val//c:pt/c:v", NS)]
            pairs = [f"{c}={v}" if c else v for c, v in zip(categories or [""] * len(values), values)]
            if pairs:
                lines.append(f"{name or '系列'}：" + "，".join(pairs))
        return lines

    def _diagram_lines(self, rel_id: str) -> list[str]:
        target = self.package.rels(self.part).get(rel_id)
        root = self.package.xml(target[1]) if target else None
        if root is None:
            return []
        return [text for text in (_clean(t.text or "") for t in root.iterfind(".//a:t", NS)) if text]

    def _picture(self, node: etree._Element, transform: _Transform, shapes: list[Shape]) -> None:
        blip = node.find("p:blipFill/a:blip", NS)
        target = self.package.rels(self.part).get(blip.get(f"{{{R}}}embed") or "") if blip is not None else None
        x, y, w, h = transform.box(*(_xfrm_box(node.find("p:spPr/a:xfrm", NS)) or (0, 0, 0, 0)))
        shapes.append(Shape(x, y, w, h, kind="picture", image=target[1] if target else ""))

    def _notes(self) -> str:
        name = self.package.related(self.part, "notesSlide")
        root = self.package.xml(name) if name else None
        if root is None:
            return ""
        texts = []
        for shape in root.iterfind("p:cSld/p:spTree//p:sp", NS):
            placeholder = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
            body = shape.find("p:txBody", NS)
            if placeholder is not None and placeholder.get("type") == "body" and body is not None:
                texts.append("\n".join(p.text for p in _paragraphs(body) if p.text.strip()))
        return _clean("\n".join(texts))


def _reading_order(slide: Slide, height: int) -> None:
    band = max(1, int(height * ROW_TOLERANCE))
    slide.shapes.sort(key=lambda shape: (shape.y // band, shape.x))


def _slide_title(slide: Slide) -> str:
    """The title placeholder; many decks draw titles as plain text boxes, so otherwise the
    largest short text in the top half, or the first short text on the slide."""
    for shape in slide.shapes:
        if shape.title and shape.text.strip():
            return _clean(shape.text).replace("\n", " ")[:80]
    best, best_size = "", 0.0
    first = ""
    for shape in slide.shapes:
        if shape.kind != "text":
            continue
        text = _clean(shape.text).replace("\n", " ")
        if not (2 <= len(text) <= 60) or text.isdigit() or re.fullmatch(r"[\d.\s]+", text):
            continue
        first = first or text
        size = max((run.size for p in shape.paragraphs for run in p.runs), default=0.0)
        if size > best_size:
            best, best_size = text, size
    return best or first


def read_deck(path: str, pause: Callable[[], None] | None = None) -> Deck:
    """A deck's slides in presentation order; a file that is not a deck raises ``ValueError``.

    ``pause()`` runs before each slide is read (the library waits there while a question
    is answered)."""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ValueError("这个 PPT 无法读取（可能有密码保护或已损坏）") from exc
    with archive:
        package = _Package(archive)
        presentation = "ppt/presentation.xml"
        try:
            root = package.xml(presentation)
            if root is None:
                raise ValueError("这个 PPT 无法读取（缺少演示文稿内容）")
            size = root.find("p:sldSz", NS)
            width = int(size.get("cx") or 12192000) if size is not None else 12192000
            height = int(size.get("cy") or 6858000) if size is not None else 6858000
            rels = package.rels(presentation)
            slides: list[Slide] = []
            for item in root.iterfind("p:sldIdLst/p:sldId", NS):
                target = rels.get(item.get(f"{{{R}}}id") or "")
                if target:
                    if pause:
                        pause()
                    slide = _SlideReader(package, target[1], len(slides) + 1).read()
                    _reading_order(slide, height)
                    slides.append(slide)
        except etree.XMLSyntaxError as exc:
            raise ValueError("这个 PPT 无法读取（内容已损坏）") from exc
    return Deck(width, height, slides)


def parse_pptx(path: str, pause: Callable[[], None] | None = None) -> tuple[list[DocumentUnit], dict[str, Any]]:
    """One unit per slide, located ``slide:N``; notes follow the slide's own text."""
    deck = read_deck(path, pause)
    units: list[DocumentUnit] = []
    picture_only: list[int] = []
    for slide in deck.slides:
        text = _clean(slide.body)
        if not text and slide.pictures:
            picture_only.append(slide.number)
        if slide.notes:
            text = f"{text}\n\n备注：{slide.notes}".strip()
        if text:
            units.append(DocumentUnit(
                text=text, heading=slide.title or f"第 {slide.number} 张", locator=f"slide:{slide.number}",
            ))
    metadata = {
        "slides": len(deck.slides),
        "pictures": sum(slide.pictures for slide in deck.slides),
        "picture_only_slides": picture_only,
        "hidden_slides": [slide.number for slide in deck.slides if slide.hidden],
    }
    return units, metadata


def deck_preview(path: str) -> dict[str, Any]:
    """What the reader draws without PowerPoint: each slide's shapes, as shares of the slide."""
    deck = read_deck(path)
    slides = []
    for slide in deck.slides:
        shapes = []
        for shape in slide.shapes:
            item: dict[str, Any] = {
                "kind": shape.kind,
                "x": round(shape.x / deck.width, 5), "y": round(shape.y / deck.height, 5),
                "w": round(shape.w / deck.width, 5), "h": round(shape.h / deck.height, 5),
            }
            if shape.kind == "picture":
                # EMF and TIFF pictures cannot be shown by a browser.
                if not shape.image.lower().endswith(BROWSER_IMAGES):
                    continue
                item["image"] = shape.image
            elif shape.kind == "table":
                item["rows"] = shape.rows
            elif shape.kind in ("chart", "diagram"):
                item["lines"] = shape.lines
            else:
                item["paragraphs"] = [
                    {"align": p.align, "runs": [
                        {"text": run.text, "size": run.size, "bold": run.bold, "color": run.color} for run in p.runs
                    ]}
                    for p in shape.paragraphs
                ]
            shapes.append(item)
        slides.append({"number": slide.number, "hidden": slide.hidden, "notes": slide.notes, "shapes": shapes})
    return {"aspect": round(deck.width / deck.height, 5), "width_pt": round(deck.width / 12700, 1), "slides": slides}


def media_part(path: str, part: str) -> bytes:
    """One picture of a deck; only names a slide refers to under ``ppt/media/`` are served."""
    if not part.startswith("ppt/media/") or ".." in part:
        raise KeyError(part)
    with zipfile.ZipFile(path) as archive:
        return archive.read(part)


def text_size_estimate(path: str) -> int:
    """Rough characters of text from the slides' XML sizes (75 to 230 bytes of XML per character in real decks)."""
    with zipfile.ZipFile(path) as archive:
        return sum(
            info.file_size for info in archive.infolist()
            if re.fullmatch(r"ppt/(?:slides|notesSlides)/[^/]+\.xml", info.filename)
        ) // 100
