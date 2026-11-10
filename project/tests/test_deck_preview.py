"""The reader for PowerPoint: PDF through local PowerPoint, the drawn fallback, and leaving the user's PowerPoint alone."""

import json
from types import SimpleNamespace
import zipfile

from fastapi import HTTPException
import pytest

from api import main
from core.rag_system import RAGSystem
from ingestion import office
from library import FileRegistry
from library.preview import (
    ApplicationBusy,
    PowerPointPdfConverter,
    PreviewPrefetcher,
    preview_kind,
    preview_plan,
)
from test_slides import make_deck, slide_xml, text_box


@pytest.mark.parametrize("ext,powerpoint,renderer,fallback", [
    (".pptx", True, "slides-pdf", "slides"),
    (".pptx", False, "slides", "slides"),
    (".ppt", True, "slides-pdf", "external"),
    (".ppt", False, "external", "external"),
])
def test_plan_for_decks(ext, powerpoint, renderer, fallback):
    plan = preview_plan(ext, True, False, powerpoint)
    assert (plan["renderer"], plan["fallback"]) == (renderer, fallback)


def test_a_deck_without_its_file_shows_the_parsed_text():
    assert preview_plan(".pptx", False, False, True)["renderer"] == "parsed"


def _converter(tmp_path, engine):
    return PowerPointPdfConverter(tmp_path / "cache", engine=engine, available=lambda: True, kill=lambda pid: None)


def _deck(tmp_path, name="a.pptx"):
    path = tmp_path / name
    make_deck(path, [slide_xml(text_box("内容"))])
    return path


def test_conversion_is_cached_like_word(tmp_path):
    calls = []

    def engine(source, target, on_pid):
        calls.append(source)
        with open(target, "wb") as out:
            out.write(b"%PDF")

    converter = _converter(tmp_path, engine)
    deck = _deck(tmp_path)
    pdf, hit = converter.convert(deck)
    assert pdf.read_bytes() == b"%PDF" and hit is False
    assert converter.convert(deck)[1] is True
    assert len(calls) == 1


def test_a_failure_is_remembered_and_names_powerpoint(tmp_path):
    def engine(source, target, on_pid):
        raise RuntimeError("COM error")

    converter = _converter(tmp_path, engine)
    deck = _deck(tmp_path)
    with pytest.raises(Exception, match="PowerPoint 无法转换"):
        converter.convert(deck)
    assert "PowerPoint 无法转换" in converter.failure(deck)


def test_powerpoint_being_open_is_not_held_against_the_file(tmp_path):
    attempts = []

    def engine(source, target, on_pid):
        attempts.append(1)
        if len(attempts) == 1:
            raise office.PowerPointBusy("PowerPoint 正在使用中，关闭后再试")
        with open(target, "wb") as out:
            out.write(b"%PDF")

    converter = _converter(tmp_path, engine)
    deck = _deck(tmp_path)
    with pytest.raises(ApplicationBusy) as error:
        converter.convert(deck)
    assert error.value.status == 409
    assert converter.failure(deck) is None
    assert converter.convert(deck)[0].read_bytes() == b"%PDF"


def test_busy_is_a_timeout_so_the_library_retries_the_file():
    from library.service import _transient

    assert _transient(office.PowerPointBusy("x"))


def test_prefetch_takes_only_its_own_kinds(tmp_path):
    deck, word = _deck(tmp_path), tmp_path / "b.docx"
    word.write_bytes(b"x")
    prefetcher = PreviewPrefetcher(
        _converter(tmp_path, lambda s, t, p: None), {"d": deck, "w": word}.get, kinds=("pptx", "ppt"),
    )
    assert prefetcher._enqueue("w", urgent=False) == "skip"
    assert prefetcher._enqueue("d", urgent=False) == "queued"
    assert preview_kind(".PPTX") == "pptx"


# ---------------- leaving the user's PowerPoint alone ----------------


def _fake_com(monkeypatch, *, running_before, running_after, presentations_left=0):
    """pywin32 stand-ins recording what would have been done to PowerPoint."""
    done = SimpleNamespace(quit=0, closed=0, killed=[], opened=[])

    class Presentation:
        def Close(self):
            done.closed += 1

    class Decks:
        Count = presentations_left

        def Open(self, **kwargs):
            done.opened.append(kwargs)
            return Presentation()

    class App:
        Presentations = Decks()
        Version = "16"

        def Quit(self):
            done.quit += 1

    states = iter([running_before, running_after])
    monkeypatch.setattr(office, "process_ids", lambda image: next(states, running_after))
    monkeypatch.setattr(office, "process_alive", lambda pid: False)
    monkeypatch.setattr(office, "kill_if_hidden", done.killed.append)
    pythoncom = pytest.importorskip("pythoncom")  # pywin32 is Windows only; so is PowerPoint
    import pywintypes
    import win32com.client.dynamic

    monkeypatch.setattr(pythoncom, "CoInitialize", lambda: None)
    monkeypatch.setattr(pythoncom, "CoUninitialize", lambda: None)
    monkeypatch.setattr(pythoncom, "CoCreateInstanceEx", lambda *a: [object()])
    monkeypatch.setattr(pywintypes, "IID", lambda name: name)
    monkeypatch.setattr(win32com.client.dynamic, "Dispatch", lambda dispatch: App())
    return done


def test_nothing_starts_while_the_user_has_powerpoint_open(monkeypatch):
    done = _fake_com(monkeypatch, running_before={111}, running_after={111})
    with pytest.raises(office.PowerPointBusy):
        office._with_powerpoint("a.pptx", lambda pid: None, lambda presentation: None)
    assert done.opened == [] and done.quit == 0 and done.killed == []


def test_our_instance_is_quit_and_the_file_opened_read_only_without_a_window(monkeypatch):
    done = _fake_com(monkeypatch, running_before=set(), running_after={222})
    pids = []
    office._with_powerpoint("a.pptx", pids.append, lambda presentation: None)
    assert pids == [222] and done.quit == 1 and done.closed == 1
    opened = done.opened[0]
    assert opened["ReadOnly"] is True and opened["WithWindow"] is False
    # A protected deck fails on the dummy password instead of asking.
    assert opened["FileName"].startswith("a.pptx::")


def test_a_deck_the_user_opened_meanwhile_keeps_powerpoint_running(monkeypatch):
    done = _fake_com(monkeypatch, running_before=set(), running_after={222}, presentations_left=1)
    office._with_powerpoint("a.pptx", lambda pid: None, lambda presentation: None)
    assert done.closed == 1 and done.quit == 0 and done.killed == []


def test_a_window_on_screen_stops_the_kill(monkeypatch):
    killed = []
    monkeypatch.setattr(office, "kill_process", killed.append)
    monkeypatch.setattr(office, "has_visible_window", lambda pid: True)
    office.kill_if_hidden(5)
    assert killed == []
    monkeypatch.setattr(office, "has_visible_window", lambda pid: False)
    office.kill_if_hidden(5)
    assert killed == [5]


def test_no_process_is_matched_by_image_name_for_killing():
    source = open(office.__file__, encoding="utf-8").read()
    assert "taskkill" not in source and "/IM" not in source


# ---------------- the API ----------------


@pytest.fixture
def api(monkeypatch, tmp_path):
    settings = SimpleNamespace(parsed_dir=tmp_path / "parsed", parent_store_dir=tmp_path / "parents", upload_dir=tmp_path / "uploads")
    settings.parsed_dir.mkdir()
    settings.upload_dir.mkdir()
    registry = FileRegistry(tmp_path / "registry.sqlite3")
    watched = tmp_path / "watched"
    watched.mkdir()
    root = registry.add_root(str(watched), None)
    picture = (
        '<p:pic><p:nvPicPr><p:cNvPr id="2" name="p"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>'
        '<p:blipFill><a:blip r:embed="rId9"/></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="6000" cy="3000"/></a:xfrm></p:spPr></p:pic>'
    )
    deck = watched / "汇报.pptx"
    make_deck(deck, [slide_xml(picture + text_box("项目背景", y=4000))], extra={"ppt/media/image1.png": b"PNGDATA"},
              rels={1: [("rId9", "image", "../media/image1.png")]})
    legacy = watched / "旧.ppt"
    legacy.write_bytes(b"binary")
    note = watched / "说明.md"
    note.write_text("内容", encoding="utf-8")
    records = {}
    for path in (deck, legacy, note):
        record = registry.insert_file(str(path), root_id=root["root_id"])
        records[path.suffix] = record["file_id"]
        (settings.parsed_dir / f"{record['file_id']}.json").write_text(
            json.dumps({"doc_id": record["file_id"], "source": record["name"], "file_type": path.suffix, "units": []}),
            encoding="utf-8",
        )
    rag = RAGSystem(settings)
    rag.library = SimpleNamespace(registry=registry)
    monkeypatch.setattr(main, "system", rag)
    monkeypatch.setattr(main, "_slides_converter", PowerPointPdfConverter(tmp_path / "cache", available=lambda: False))
    return records


def test_without_powerpoint_a_deck_is_drawn_and_a_legacy_deck_offers_opening_it(api):
    assert main.get_document(api[".pptx"])["preview"]["renderer"] == "slides"
    assert main.get_document(api[".ppt"])["preview"]["renderer"] == "external"
    with pytest.raises(HTTPException) as error:
        main.document_as_pdf(api[".pptx"], wait=0)
    assert error.value.status_code == 409 and "PowerPoint" in error.value.detail


def test_with_powerpoint_a_deck_converts(api, monkeypatch, tmp_path):
    def engine(source, target, on_pid):
        with open(target, "wb") as out:
            out.write(b"%PDF")

    monkeypatch.setattr(main, "_slides_converter", PowerPointPdfConverter(tmp_path / "cache2", engine=engine, available=lambda: True))
    assert main.get_document(api[".pptx"])["preview"]["renderer"] == "slides-pdf"
    assert str(main.document_as_pdf(api[".pptx"], wait=0).path).endswith(".pdf")


def test_slides_endpoint_and_its_pictures(api):
    deck = main.document_slides(api[".pptx"])
    assert deck["slides"][0]["number"] == 1
    kinds = [shape["kind"] for shape in deck["slides"][0]["shapes"]]
    assert kinds == ["picture", "text"]
    media = main.document_slide_media(api[".pptx"], part="ppt/media/image1.png")
    assert media.body == b"PNGDATA" and media.media_type == "image/png"


@pytest.mark.parametrize("part", ["ppt/slides/slide1.xml", "../../secret.png", "ppt/media/none.png", "ppt/media/../presentation.xml"])
def test_slide_media_serves_nothing_else(api, part):
    with pytest.raises(HTTPException) as error:
        main.document_slide_media(api[".pptx"], part=part)
    assert error.value.status_code == 404


def test_slides_endpoint_refuses_other_files(api):
    with pytest.raises(HTTPException) as error:
        main.document_slides(api[".md"])
    assert error.value.status_code == 400
    with pytest.raises(HTTPException) as error:
        main.document_slides(api[".ppt"])
    assert error.value.status_code == 400


def test_a_broken_deck_says_why(api, monkeypatch, tmp_path):
    bad = tmp_path / "bad.pptx"
    bad.write_bytes(b"nope")
    monkeypatch.setattr(main, "_original_or_404", lambda doc_id: bad)
    with pytest.raises(HTTPException) as error:
        main.document_slides("x")
    assert error.value.status_code == 422 and "密码保护或已损坏" in error.value.detail


def test_only_the_slides_prefetcher_sees_decks(api, monkeypatch, tmp_path):
    def engine(source, target, on_pid):
        with open(target, "wb") as out:
            out.write(b"%PDF")

    monkeypatch.setattr(main, "_slides_converter", PowerPointPdfConverter(tmp_path / "cache3", engine=engine, available=lambda: True))
    monkeypatch.setattr(main, "_prefetcher", None)
    monkeypatch.setattr(main, "_slides_prefetcher", None)
    monkeypatch.setattr(main, "_word_converter", None)
    monkeypatch.setattr(main.settings, "data_dir", tmp_path / "data", raising=False)
    try:
        states = main.prepare_documents(main.PrepareRequest(doc_ids=[api[".pptx"], api[".md"]]))["states"]
        assert states[api[".pptx"]] in ("queued", "ready")
        assert states[api[".md"]] == "skip"
    finally:
        main.preview_prefetcher().stop()
        main.slides_prefetcher().stop()


def test_zip_stays_closed_after_reading(tmp_path):
    deck = _deck(tmp_path)
    from ingestion.slides import parse_pptx

    parse_pptx(str(deck))
    deck.unlink()  # on Windows this fails if a handle is left open
    assert not deck.exists()
    assert zipfile.is_zipfile(_deck(tmp_path, "b.pptx"))
