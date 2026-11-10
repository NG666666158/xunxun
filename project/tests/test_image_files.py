"""Image files as library documents: decoding, scaling, the three reading modes and the cloud request."""

from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from ingestion import images
from ingestion.figures import IMAGE_PROMPT, CloudReader, Figure
from ingestion.parsers import MultiFormatParser, parser_version

PIL = pytest.importorskip("PIL.Image")


def _settings(tmp_path, **extra):
    return SimpleNamespace(
        data_dir=tmp_path / "data", max_upload_mb=20, max_local_file_mb=20, max_local_text_mb=20, image_max_side=2000, image_understanding="off",
        local_formula=False, vision_concurrency=6, **extra,
    )


def _image(path: Path, size=(3000, 1500), mode="RGB", color=(255, 255, 255), **save):
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.new(mode, size, color).save(path, **save)
    return path


class FakeReader:
    mode = "local"
    model = "fake"
    workers = 1

    def __init__(self, text="发票号码 12345", fail=False):
        self.text = text
        self.fail = fail
        self.figures: list[Figure] = []

    def read(self, figure):
        self.figures.append(figure)
        if self.fail:
            raise TimeoutError("vision model timed out")
        return self.text


def test_load_pages_scales_the_long_side_and_keeps_jpeg(tmp_path):
    png = _image(tmp_path / "wide.png", (3000, 1500), dpi=(300, 300))  # 300 dpi: page points != pixels
    (data, mime), = images.load_pages(png, 2000)
    pix = pymupdf.Pixmap(data)
    assert mime == "image/png" and (pix.width, pix.height) == (2000, 1000)

    jpg = _image(tmp_path / "photo.jpg", (4000, 3000))
    (data, mime), = images.load_pages(jpg, 2000)
    assert mime == "image/jpeg" and max(pymupdf.Pixmap(data).width, pymupdf.Pixmap(data).height) == 2000

    small = _image(tmp_path / "small.bmp", (640, 480))
    (data, _), = images.load_pages(small, 2000)
    assert pymupdf.Pixmap(data).width == 640  # never enlarged


def test_transparent_png_is_painted_on_white_and_tiff_pages_are_kept(tmp_path):
    clear = _image(tmp_path / "clear.png", (50, 50), mode="RGBA", color=(0, 0, 0, 0))
    (data, _), = images.load_pages(clear, 2000)
    assert pymupdf.Pixmap(data).pixel(10, 10)[:3] == (255, 255, 255)  # black-on-transparent stays readable

    frames = [PIL.new("RGB", (800, 600), (0, 0, 255)), PIL.new("RGB", (400, 300), (0, 255, 0))]
    frames[0].save(tmp_path / "scan.tif", save_all=True, append_images=frames[1:])
    pages = images.load_pages(tmp_path / "scan.tif", 2000)
    assert len(pages) == 2 and pymupdf.Pixmap(pages[1][0]).width == 400

    webp = _image(tmp_path / "pic.webp", (2400, 1200))  # PyMuPDF cannot decode WebP: Pillow does
    (data, mime), = images.load_pages(webp, 2000)
    assert mime == "image/png" and pymupdf.Pixmap(data).width == 2000


def test_name_mode_indexes_name_folder_and_date_only(tmp_path):
    path = _image(tmp_path / "2024 三亚旅行" / "IMG_2034.jpg", (100, 100))
    reader = FakeReader()
    units, metadata = images.parse_image(path, "name", reader, _settings(tmp_path))
    assert reader.figures == []  # nothing is read, nothing uploaded
    text = units[0].text
    assert "IMG_2034.jpg" in text and "2024 三亚旅行" in text and "日期：" in text
    assert metadata["image_mode"] == "name"


def test_exif_date_is_preferred_over_the_file_time(tmp_path):
    exif = PIL.Exif()
    exif[306] = "2019:07:01 10:20:30"
    path = _image(tmp_path / "old.jpg", (64, 64), exif=exif)
    assert images.taken_at(path).year == 2019
    assert "2019-07-01" in images.header(path)


def test_local_mode_reads_text_and_falls_back_to_the_name_without_ocr(tmp_path, monkeypatch):
    path = _image(tmp_path / "票据" / "发票.png", (1200, 800))
    monkeypatch.setattr(images.ocr_runtime, "activate", lambda data_dir, component="ocr": True)
    reader = FakeReader()
    units, metadata = images.parse_image(path, "local", reader, _settings(tmp_path))
    assert "〔图中文字〕" in units[0].text and "发票号码 12345" in units[0].text
    assert reader.figures[0].kind == "image" and reader.figures[0].context == "发票.png"
    assert metadata["text_chars"] == len("发票号码 12345") and "notice" not in metadata

    # Local OCR not downloaded yet: findable by name now, read once the component arrives.
    monkeypatch.setattr(images.ocr_runtime, "activate", lambda data_dir, component="ocr": False)
    units, metadata = images.parse_image(path, "local", FakeReader(), _settings(tmp_path))
    assert metadata["notice"] == "ocr_unavailable" and "发票.png" in units[0].text


def test_a_failed_cloud_reading_raises_so_the_file_is_retried(tmp_path):
    path = _image(tmp_path / "chart.jpg", (500, 400))
    reader = FakeReader(fail=True)
    with pytest.raises(TimeoutError):
        images.parse_image(path, "cloud", reader, _settings(tmp_path))
    assert reader.figures[0].mime == "image/jpeg"


def test_parser_reads_image_files_in_the_files_own_mode(tmp_path, monkeypatch):
    path = _image(tmp_path / "白板.png", (800, 600))
    used = []
    monkeypatch.setattr("ingestion.parsers.build_reader", lambda settings, mode=None: used.append(mode) or FakeReader("白板 讨论纪要"))
    monkeypatch.setattr(images.ocr_runtime, "activate", lambda data_dir, component="ocr": True)
    parser = MultiFormatParser(_settings(tmp_path))
    parsed = parser.parse(path, doc_id="a" * 32, image_mode="cloud")
    assert used == ["cloud"] and "白板 讨论纪要" in parsed.units[0].text
    assert parsed.file_type == ".png"
    versions = {mode: parser_version(path, _settings(tmp_path), mode) for mode in images.IMAGE_MODES}
    assert len(set(versions.values())) == 3  # switching a file's mode re-reads it
    assert parser_version(path, _settings(tmp_path)) == versions["local"]  # the default


def test_reading_cache_works_past_260_characters(tmp_path):
    from ingestion.figures import CachedReader

    reader = FakeReader("缓存的文字")
    deep = tmp_path / ("数据" * 40) / ("缓存" * 40) / "cache" / "figures"
    cached = CachedReader(reader, deep)
    figure = Figure(page=1, bbox=(0, 0, 0, 0), kind="image", image=b"same image")
    assert len(str(deep)) + 70 > 260
    assert cached.read(figure) == cached.read(figure) == "缓存的文字"
    assert len(reader.figures) == 1  # the second reading came from the cache


def test_cloud_reader_sends_the_image_with_its_type_and_the_image_prompt():
    reader = CloudReader.__new__(CloudReader)
    calls = []

    def create(**request):
        calls.append(request)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="截图：报销流程\n第一步 提交"))])

    reader.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    reader.model, reader.ocr_model, reader.workers = "vl", "ocr", 1
    text = reader.read(Figure(page=1, bbox=(0, 0, 0, 0), kind="image", context="流程.jpg", image=b"jpg", mime="image/jpeg"))
    content = calls[0]["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1]["text"] == IMAGE_PROMPT.format(name="流程.jpg") and calls[0]["model"] == "vl"
    assert text.startswith("截图：报销流程")
