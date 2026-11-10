# Attribution and modifications

This repository is based on Giovanni Pasqualino's MIT-licensed
[`agentic-rag-for-dummies`](https://github.com/GiovanniPasq/agentic-rag-for-dummies).
The original copyright notice is preserved in [`LICENSE`](LICENSE).

The enterprise-oriented version in this repository independently adds or
rewrites the following areas:

- PDF, DOCX, XLSX, Markdown and TXT ingestion;
- Parent-Child chunking with persistent parent recovery;
- Milvus Dense + BM25 retrieval, RRF fusion and BGE reranking;
- explicit LangGraph routing, bounded retries and evidence checking;
- allow-listed structured-data and mock business tools;
- FastAPI, validated SSE output and a web workbench (plain HTML/CSS/JS);
- deterministic synthetic-data generators and frozen evaluation protocols;
- Docker Compose deployment, regression tests and benchmark reporting.

All companies, people, product identifiers, prices and internal policies used
by the demo or benchmark generators are fictional.

## Third-party components shipped with the desktop app

The reader panel's original-file view bundles these libraries locally (no CDN; the
desktop webview's CSP only allows its own files). Each folder under
`project/ui/static/vendor/` is named after the pinned version and keeps the upstream
license file.

| Component | Version | License | Used for |
|---|---|---|---|
| [PDF.js](https://github.com/mozilla/pdf.js) (`pdfjs-dist`: `pdf.min.mjs`, `pdf.worker.min.mjs`, `pdf_viewer.mjs/.css`, CMaps, standard fonts) | 4.10.38 | Apache-2.0 | PDF pages, zoom, find, highlighting; Word files converted to PDF |
| [docx-preview](https://github.com/VolodymyrBaydalka/docxjs) | 0.4.1 | Apache-2.0 | Word preview when Word is not installed or conversion fails |
| [JSZip](https://github.com/Stuk/jszip) | 3.10.1 | MIT (dual-licensed MIT or GPL-3.0-or-later; used under MIT) | Required by docx-preview |
| [SheetJS Community Edition](https://sheetjs.com) (`xlsx.core.min.js`) | 0.20.3 | Apache-2.0 | Excel worksheets as tables |

The Python backend additionally uses [pywin32](https://github.com/mhammond/pywin32)
(PSF-2.0) on Windows to drive the locally installed Microsoft Word through COM when
converting Word documents to PDF for preview. Microsoft Word itself is not shipped.
