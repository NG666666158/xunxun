"""A deterministic test library for import-speed runs, made only from public text.

The text is the CRUD-RAG news corpus (IAAR-Shanghai/CRUD_RAG, 80000_docs; fetched by
evaluation/suite/fetch.py into evaluation/external/crud_rag). Articles are joined into
documents of a long-tailed size and written in the formats people keep in a work folder:

    python -I evaluation/import_speed_v1/make_corpus.py <crud_rag/80000_docs> <out folder> --files 400

Mix (by file count): Markdown 15 %, text 8 %, Word 25 %, Excel 10 %, PowerPoint 8 %, text PDF 22 %,
scanned PDF 4 %, images 8 %. Sizes: lognormal, median 4,000 characters (mean about 9,000), from
300 to 150,000. ``--seed`` changes the draw; the same arguments always give the same files.
``<out>.manifest.json`` (next to the folder) lists every file with its type and character count.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import random
import re
import zipfile

MIX = (
    ("md", 0.15), ("txt", 0.08), ("docx", 0.25), ("xlsx", 0.10), ("pptx", 0.08),
    ("pdf", 0.22), ("scan", 0.04), ("image", 0.08),
)
FOLDERS = ("行业动态", "市场分析", "项目资料/2023", "项目资料/2024", "会议纪要", "技术笔记", "归档/旧资料")
PAGE_CHARS = 700


def articles(source: Path, needed: int) -> list[str]:
    """News articles (date prefix removed), in file order, until ``needed`` characters."""
    found: list[str] = []
    total = 0
    for part in sorted(source.iterdir()):
        for line in part.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = line.split("正文：", 1)[-1].strip()
            if len(text) < 80:
                continue
            found.append(text)
            total += len(text)
            if total >= needed:
                return found
    return found


class Text:
    def __init__(self, items: list[str]):
        self.items = items
        self.at = 0

    def take(self, chars: int) -> list[str]:
        """Whole articles adding up to about ``chars`` (the last one cut)."""
        out: list[str] = []
        size = 0
        while size < chars:
            item = self.items[self.at % len(self.items)]
            self.at += 1
            piece = item[: chars - size] if size + len(item) > chars * 1.3 else item
            out.append(piece)
            size += len(piece)
        return out


def title(text: str) -> str:
    head = re.split(r"[，。！？；：,.!?;:\s]", text, maxsplit=1)[0]
    return re.sub(r'[\\/:*?"<>|]', "", head)[:18] or "资料"


def sections(parts: list[str]) -> list[tuple[str, str]]:
    return [(f"{n}. {title(p)}", p) for n, p in enumerate(parts, 1)]


def write_md(path: Path, name: str, parts: list[str]) -> None:
    body = [f"# {name}", ""]
    for heading, text in sections(parts):
        body += [f"## {heading}", "", text, ""]
    path.write_text("\n".join(body), encoding="utf-8")


def write_txt(path: Path, name: str, parts: list[str]) -> None:
    path.write_text(name + "\n\n" + "\n\n".join(parts), encoding="utf-8")


def write_docx(path: Path, name: str, parts: list[str]) -> None:
    from docx import Document

    document = Document()
    document.add_heading(name, level=1)
    for heading, text in sections(parts):
        document.add_heading(heading, level=2)
        for paragraph in re.split(r"(?<=。)", text):
            if paragraph.strip():
                document.add_paragraph(paragraph)
    document.save(path)


def write_xlsx(path: Path, name: str, parts: list[str]) -> None:
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.title = "明细"
    sheet.append(["序号", "标题", "摘要", "字数", "分类"])
    for n, text in enumerate(parts, 1):
        for k, sentence in enumerate(s for s in re.split(r"(?<=。)", text) if s.strip()):
            sheet.append([f"{n}-{k + 1}", title(text), sentence[:160], len(sentence), FOLDERS[n % len(FOLDERS)]])
    book.save(path)


NS = (
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
    'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
)
RELS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _box(text: str, y: int, size: int = 0) -> str:
    props = f'<a:rPr sz="{size}" b="1"/>' if size else ""
    paragraphs = "".join(f"<a:p><a:r>{props}<a:t>{_escape(line)}</a:t></a:r></a:p>" for line in text.split("\n"))
    return (
        '<p:sp><p:nvSpPr><p:cNvPr id="1" name="s"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
        f'<p:spPr><a:xfrm><a:off x="0" y="{y}"/><a:ext cx="12000" cy="1000"/></a:xfrm></p:spPr>'
        f"<p:txBody><a:bodyPr/>{paragraphs}</p:txBody></p:sp>"
    )


def write_pptx(path: Path, name: str, parts: list[str]) -> None:
    slides = []
    for heading, text in sections(parts):
        for k in range(0, len(text), 400):
            bullets = "\n".join(s for s in re.split(r"(?<=。)", text[k : k + 400]) if s.strip())
            slides.append(f'<p:sld {NS}><p:cSld><p:spTree>{_box(heading, 0, 3200)}{_box(bullets, 1200)}</p:spTree></p:cSld></p:sld>')
    names = [f"slide{i + 1}.xml" for i in range(len(slides))]
    ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{i + 1}"/>' for i in range(len(slides)))
    links = "".join(f'<Relationship Id="rId{i + 1}" Type="{OFFICE}/slide" Target="slides/{n}"/>' for i, n in enumerate(names))
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "ppt/presentation.xml",
            f'<p:presentation {NS}><p:sldSz cx="12000" cy="6750"/><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>',
        )
        archive.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships xmlns="{RELS}">{links}</Relationships>')
        for slide_name, xml in zip(names, slides):
            archive.writestr(f"ppt/slides/{slide_name}", xml)


def _pages(name: str, parts: list[str]) -> list[str]:
    flat = name + "\n" + "\n".join(f"{h}\n{t}" for h, t in sections(parts))
    return [flat[k : k + PAGE_CHARS] for k in range(0, len(flat), PAGE_CHARS)]


def _text_pdf(pages: list[str]):
    import pymupdf

    document = pymupdf.open()
    for text in pages:
        page = document.new_page(width=595, height=842)
        page.insert_textbox(pymupdf.Rect(56, 56, 539, 786), text, fontname="china-s", fontsize=11)
    return document


def write_pdf(path: Path, name: str, parts: list[str]) -> None:
    document = _text_pdf(_pages(name, parts))
    document.save(path, garbage=3, deflate=True)


def write_scan(path: Path, name: str, parts: list[str]) -> None:
    """Pages as pictures only (no text layer), like a scanner's output."""
    import pymupdf

    source = _text_pdf(_pages(name, parts))
    scanned = pymupdf.open()
    for page in source:
        picture = page.get_pixmap(dpi=110, colorspace=pymupdf.csGRAY)
        target = scanned.new_page(width=page.rect.width, height=page.rect.height)
        target.insert_image(target.rect, stream=picture.tobytes("png"))
    scanned.save(path, garbage=3, deflate=True)


def write_image(path: Path, name: str, parts: list[str]) -> None:
    """A photographed / screenshotted page: one page of text as PNG or JPEG."""
    source = _text_pdf(_pages(name, parts)[:1])
    picture = source[0].get_pixmap(dpi=100)
    path.write_bytes(picture.tobytes("jpeg" if path.suffix == ".jpg" else "png"))


WRITERS = {
    "md": (write_md, ".md"), "txt": (write_txt, ".txt"), "docx": (write_docx, ".docx"),
    "xlsx": (write_xlsx, ".xlsx"), "pptx": (write_pptx, ".pptx"), "pdf": (write_pdf, ".pdf"),
    "scan": (write_scan, ".pdf"), "image": (write_image, ".png"),
}


def size_for(kind: str, rng: random.Random, median: int = 4000, largest: int = 150000) -> int:
    if kind == "image":
        return PAGE_CHARS
    if kind == "scan":
        return PAGE_CHARS * rng.randint(2, 10)
    chars = int(math.exp(rng.gauss(math.log(median), 1.25)))
    if kind in ("pptx", "xlsx"):
        chars = min(chars, 40000)
    return max(300, min(chars, largest))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="crud_rag/80000_docs")
    parser.add_argument("out", type=Path)
    parser.add_argument("--files", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--offset", type=int, default=0, help="skip this many characters of news (a second corpus)")
    parser.add_argument("--only", default="", help="one kind only, e.g. image (for the recognition runs)")
    parser.add_argument("--text-only", action="store_true", help="no images or scans")
    parser.add_argument("--median", type=int, default=4000, help="median characters per document")
    parser.add_argument("--largest", type=int, default=150000)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    mix = ((args.only, 1.0),) if args.only else MIX
    if args.text_only:
        mix = tuple((kind, share / 0.88) for kind, share in mix if kind not in ("image", "scan"))
    kinds = [kind for kind, share in mix for _ in range(round(share * args.files))]
    while len(kinds) < args.files:
        kinds.append("docx")
    kinds = kinds[: args.files]
    rng.shuffle(kinds)
    sizes = [size_for(kind, rng, args.median, args.largest) for kind in kinds]
    text = Text(articles(args.source, sum(sizes) + args.offset + 10000))
    text.take(args.offset) if args.offset else None
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = []
    for number, (kind, chars) in enumerate(zip(kinds, sizes), 1):
        parts = text.take(chars)
        name = title(parts[0])
        writer, suffix = WRITERS[kind]
        if kind == "image" and number % 3 == 0:
            suffix = ".jpg"
        folder = args.out / FOLDERS[rng.randrange(len(FOLDERS))]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{number:05d}_{name}{suffix}"
        writer(path, name, parts)
        manifest.append({
            "path": str(path.relative_to(args.out)), "kind": kind, "characters": sum(map(len, parts)),
            "bytes": path.stat().st_size,
        })
    (args.out.parent / f"{args.out.name}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    by_kind: dict[str, list[int]] = {}
    for item in manifest:
        by_kind.setdefault(item["kind"], []).append(item["characters"])
    print(json.dumps({k: {"files": len(v), "characters": sum(v)} for k, v in by_kind.items()}, ensure_ascii=False))
    print("total characters", sum(item["characters"] for item in manifest))


if __name__ == "__main__":
    main()
