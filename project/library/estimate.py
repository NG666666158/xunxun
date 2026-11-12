"""How long adding a batch of files takes, for the confirmation before a big import.

Measured in evaluation/import_speed_v1/REPORT.md (400 files of public news in eight formats,
SiliconFlow bge-m3 on the account tested). The stages run side by side, so the batch takes as long
as the slowest of them, not their sum:

* embedding: the account's token limit (about 500k tokens a minute for bge-m3 on the tested
  account; ``tokens_per_minute`` is what the pacer has learned so far), after a first burst;
* requests: each takes about 0.2 s and ``workers`` files are read at once (small files);
* the processor: parsing, word segmentation and database writes, one core's worth (Python);
* recognition: images and scanned pages on ``ocr_jobs`` engines at once.

The old estimate added a fixed 1.7 s per file and 1.5 s per request one after another, and
said 7.5 hours for a folder of 9,347 files.
"""

from __future__ import annotations

import math
from typing import Any

# Children are ~1.24x the text (overlap and the "document · section" prefix) and bge-m3 counts
# ~0.57 tokens per character of Chinese: tokens sent per character of text.
TOKENS_PER_CHARACTER = 0.72
# The file entry (name, chapter titles and opening) embedded once per file.
TOKENS_PER_FILE = 150
CHARACTERS_PER_CHUNK = 520
CHILDREN_PER_REQUEST = 32
# A request of a small file's few children or a file entry: about 0.2 s (api_bench.py, small1500).
SECONDS_PER_REQUEST = 0.2
# Processor time: parsing, segmentation and writes per character, plus a little per file.
CPU_SECONDS_PER_CHARACTER = 1.8e-5
CPU_SECONDS_PER_FILE = 0.06
# Local recognition of an image full of text, per image and engine (images.py has the
# lighter per-image figures used for the choice of image mode).
OCR_SECONDS_PER_PAGE = {"local": 7.0, "cloud": 10.0}
# Setting up, scanning the folder, the last files' tail.
OVERHEAD_SECONDS = 10.0
SLACK = 1.15


def text_seconds(
    documents: int,
    characters: int,
    *,
    tokens_per_minute: float,
    burst_tokens: float = 0.0,
    workers: int = 4,
) -> dict[str, float]:
    """Seconds for the text of ``documents`` files with ``characters`` characters in all, by stage."""
    tokens = characters * TOKENS_PER_CHARACTER + documents * TOKENS_PER_FILE
    embedding = max(0.0, tokens - burst_tokens) * 60.0 / max(1.0, tokens_per_minute)
    requests = math.ceil(characters / CHARACTERS_PER_CHUNK / CHILDREN_PER_REQUEST) + 2 * documents
    return {
        "tokens": round(tokens),
        "embedding": embedding,
        "requests": requests * SECONDS_PER_REQUEST / max(1, workers),
        "processor": characters * CPU_SECONDS_PER_CHARACTER + documents * CPU_SECONDS_PER_FILE,
    }


def batch_seconds(
    documents: int,
    characters: int,
    image_seconds: float,
    scanned_pages: int = 0,
    page_mode: str = "off",
    *,
    tokens_per_minute: float,
    burst_tokens: float = 0.0,
    workers: int = 4,
    ocr_jobs: int = 2,
) -> dict[str, Any]:
    """The whole batch: text, images (``image_seconds``, already for the chosen mode) and
    scanned PDF pages read in ``page_mode``, side by side."""
    text = text_seconds(documents, characters, tokens_per_minute=tokens_per_minute,
                        burst_tokens=burst_tokens, workers=workers)
    pages = scanned_pages * OCR_SECONDS_PER_PAGE.get(page_mode, 0.0) / max(1, ocr_jobs)
    slowest = max(text["embedding"], text["requests"], text["processor"], image_seconds + pages)
    return {
        "seconds": round(OVERHEAD_SECONDS + slowest * SLACK) if documents or image_seconds or pages else 0,
        "text_seconds": round(max(text["embedding"], text["requests"], text["processor"]) * SLACK),
        "tokens": text["tokens"],
        "bound": max(
            (("embedding", text["embedding"]), ("requests", text["requests"]),
             ("processor", text["processor"]), ("recognition", image_seconds + pages)),
            key=lambda item: item[1],
        )[0],
    }
