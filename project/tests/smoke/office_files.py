"""The smallest .pptx and .xlsx the app reads, written by hand so these tests need no Office
library in the environment that runs them (the browser tests run outside the app's venv)."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import zipfile

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
PACKAGE = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
DRAWING = "http://schemas.openxmlformats.org/drawingml/2006/main"
PRESENTATION = "http://schemas.openxmlformats.org/presentationml/2006/main"
_DECK_NS = f'xmlns:a="{DRAWING}" xmlns:p="{PRESENTATION}" xmlns:r="{OFFICE}"'


def write_deck(path: Path, slides: list[str]) -> Path:
    """One text box per slide."""
    ids = "".join(f'<p:sldId id="{256 + i}" r:id="rId{i + 1}"/>' for i in range(len(slides)))
    links = "".join(
        f'<Relationship Id="rId{i + 1}" Type="{OFFICE}/slide" Target="slides/slide{i + 1}.xml"/>' for i in range(len(slides))
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "ppt/presentation.xml",
            f'<p:presentation {_DECK_NS}><p:sldSz cx="12192000" cy="6858000"/><p:sldIdLst>{ids}</p:sldIdLst></p:presentation>',
        )
        archive.writestr("ppt/_rels/presentation.xml.rels", f'<Relationships xmlns="{PACKAGE}">{links}</Relationships>')
        for number, text in enumerate(slides, start=1):
            archive.writestr(
                f"ppt/slides/slide{number}.xml",
                f'<p:sld {_DECK_NS}><p:cSld><p:spTree><p:sp><p:nvSpPr><p:cNvPr id="1" name="s"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
                '<p:spPr><a:xfrm><a:off x="914400" y="914400"/><a:ext cx="10363200" cy="1371600"/></a:xfrm></p:spPr>'
                f"<p:txBody><a:bodyPr/><a:p><a:r><a:t>{escape(text)}</a:t></a:r></a:p></p:txBody></p:sp>"
                "</p:spTree></p:cSld></p:sld>",
            )
    return path


def write_workbook(path: Path, rows: list[list[str]]) -> Path:
    """One sheet of text cells; the first row is the header."""

    def cell(column: int, row: int, value: str) -> str:
        return f'<c r="{chr(65 + column)}{row}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'

    body = "".join(
        f'<row r="{r}">' + "".join(cell(c, r, value) for c, value in enumerate(row)) + "</row>"
        for r, row in enumerate(rows, start=1)
    )
    sheet = f'<worksheet xmlns="{MAIN}"><sheetData>{body}</sheetData></worksheet>'
    book = f'<workbook xmlns="{MAIN}" xmlns:r="{OFFICE}"><sheets><sheet name="名单" sheetId="1" r:id="rId1"/></sheets></workbook>'
    types = (
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        f'<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        "</Types>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", types)
        archive.writestr(
            "_rels/.rels",
            f'<Relationships xmlns="{PACKAGE}"><Relationship Id="rId1" Type="{OFFICE}/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        )
        archive.writestr("xl/workbook.xml", book)
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            f'<Relationships xmlns="{PACKAGE}"><Relationship Id="rId1" Type="{OFFICE}/worksheet" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        archive.writestr("xl/worksheets/sheet1.xml", sheet)
    return path
