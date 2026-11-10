"use strict";

/* Reader panel, "原文" view: shows the original file so the user can tell at a glance
   whether it is the one they want. Loaded after app.js and uses its helpers (apiFetch,
   API, DESKTOP, esc, errorDetail, toast).

   Renderer per format comes from the backend's plan (library/preview.py):
     pdfjs        PDF.js viewer (vendor/, pinned) - pages, zoom, find, cited text highlighted
     word-pdf     Word on this machine converts to PDF (cached), then PDF.js
     slides-pdf   PowerPoint on this machine converts to PDF (cached), then PDF.js
     slides       a deck's slides drawn from its own XML (no PowerPoint, or the conversion failed)
     docx-preview docx rendered in the page (no Word, or the conversion failed)
     sheetjs      SheetJS, one tab per worksheet
     markdown / text / image
     external     "open with the default program"
   Every request carries the per-launch token: PDF.js via httpHeaders, everything else is
   fetched as a blob first. */
const Preview = (() => {
  const PDFJS = "vendor/pdfjs-4.10.38/";
  const SCRIPTS = {
    jszip: "vendor/jszip-3.10.1/jszip.min.js",
    docx: "vendor/docx-preview-0.4.1/docx-preview.min.js",
    xlsx: "vendor/xlsx-0.20.3/xlsx.core.min.js",
  };
  const SHEET_MAX_ROWS = 2000;
  const SHEET_MAX_COLS = 80;
  const PDF_SCAN_PAGES = 400; // looking for a quote without a page number

  const els = {
    root: document.getElementById("viewerOrig"),
    tools: document.getElementById("origTools"),
    how: document.getElementById("viewerHow"),
  };
  const loaded = new Map();
  let pdfjs = null;
  let current = null; // the shown preview: { docId, cleanup[], focus(f), dead }

  const abs = (path) => new URL(path, document.baseURI).href;

  function loadScript(src) {
    if (!loaded.has(src)) {
      loaded.set(src, new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = src;
        s.onload = resolve;
        s.onerror = () => { loaded.delete(src); reject(new Error(`无法加载组件 ${src}`)); };
        document.head.append(s);
      }));
    }
    return loaded.get(src);
  }

  function loadCss(href) {
    if (document.querySelector(`link[data-preview="${href}"]`)) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    link.dataset.preview = href;
    document.head.append(link);
  }

  function loadPdfjs() {
    pdfjs ||= (async () => {
      const lib = await import(abs(`${PDFJS}pdf.min.mjs`));
      lib.GlobalWorkerOptions.workerSrc = abs(`${PDFJS}pdf.worker.min.mjs`);
      globalThis.pdfjsLib = lib; // pdf_viewer.mjs reads the library from here
      const viewer = await import(abs(`${PDFJS}pdf_viewer.mjs`));
      loadCss(`${PDFJS}pdf_viewer.css`);
      return { lib, viewer };
    })().catch((err) => { pdfjs = null; throw err; });
    return pdfjs;
  }

  const fileUrl = (docId) => `${API}/documents/${encodeURIComponent(docId)}/file?inline=1`;
  // Image files: the original, or a PNG of a TIFF (which the webview cannot draw).
  const imageUrl = (docId) => `${API}/documents/${encodeURIComponent(docId)}/image`;
  const authHeaders = () => (DESKTOP?.token ? { "X-App-Token": DESKTOP.token } : {});

  async function fetchOk(url) {
    const res = await apiFetch(url);
    if (!res.ok) throw new Error(await errorDetail(res));
    return res;
  }

  function setHow(text, warn = false) {
    els.how.hidden = !text;
    els.how.textContent = text || "";
    els.how.classList.toggle("warn", warn);
  }

  /** 寻寻 fetching the file (app.js): the same for every format, nothing about how. */
  function loading() {
    showFetching(els.root);
  }

  // ---------------------------------------------------------------- quotes

  /** Comparable form: no whitespace, full-width folded (NFKC), case-insensitive. */
  function norm(text) {
    return String(text || "").normalize("NFKC").replace(/[\s​…]+/g, "").toLowerCase();
  }

  /** Longest stretch of `quote` found in `flat` (both normalized): [start, end) or null. */
  function findQuote(flat, quote, minimum = 8) {
    const q = norm(quote);
    if (q.length < minimum || !flat) return null;
    const key = Math.min(12, q.length);
    let best = null;
    for (let off = 0; off + key <= q.length && off < 160; off += 6) {
      let at = flat.indexOf(q.slice(off, off + key));
      while (at >= 0) {
        let start = at;
        let end = at + key;
        let qi = off + key;
        while (end < flat.length && qi < q.length && flat[end] === q[qi]) { end++; qi++; }
        let qb = off - 1;
        while (start > 0 && qb >= 0 && flat[start - 1] === q[qb]) { start--; qb--; }
        if (!best || end - start > best[1] - best[0]) best = [start, end];
        at = flat.indexOf(q.slice(off, off + key), at + 1);
      }
      if (best && best[1] - best[0] >= Math.min(40, q.length)) break;
    }
    return best && best[1] - best[0] >= minimum ? best : null;
  }

  /** Mark the elements (text layer spans, text nodes' parents) that hold the quote. */
  function markQuote(container, quote, { selector = null, cls = "quote-hit" } = {}) {
    const owners = [];
    let flat = "";
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const owner = selector ? node.parentElement.closest(selector) : node.parentElement;
      if (!owner) continue;
      const piece = norm(node.data);
      for (const ch of piece) { flat += ch; owners.push(owner); }
    }
    const found = findQuote(flat, quote);
    if (!found) return null;
    const hits = [...new Set(owners.slice(found[0], found[1]))];
    hits.forEach((el) => el.classList.add(cls));
    return hits[0];
  }

  // ---------------------------------------------------------------- PDF

  function pageRange(locator) {
    // One PDF page per slide, so a slide citation is that page.
    const m = (locator || "").match(/^(?:page|slide):(\d+)(?:-(\d+))?$/);
    return m ? [Number(m[1]), Number(m[2] || m[1])] : null;
  }

  async function renderPdf(ctx, source, focus) {
    const { lib, viewer: V } = await loadPdfjs();
    if (ctx.dead) return;
    els.root.innerHTML = '<div class="pdf-scroll"><div class="pdfViewer"></div></div>';
    const container = els.root.firstElementChild;
    const eventBus = new V.EventBus();
    const linkService = new V.PDFLinkService({ eventBus });
    linkService.externalLinkEnabled = false; // a link must not navigate the app away
    const findController = new V.PDFFindController({ eventBus, linkService });
    const pdfViewer = new V.PDFViewer({
      container,
      viewer: container.firstElementChild,
      eventBus,
      linkService,
      findController,
      textLayerMode: 1,
      annotationMode: lib.AnnotationMode.ENABLE,
      annotationEditorMode: lib.AnnotationEditorType.DISABLE,
      imageResourcesPath: abs(`${PDFJS}images/`),
      removePageBorders: false,
    });
    linkService.setViewer(pdfViewer);

    els.tools.innerHTML = `
      <div class="pdf-pager">
        <button type="button" class="icon-btn sm" data-pdf="prev" title="上一页" aria-label="上一页"><svg viewBox="0 0 24 24"><path d="M15 6l-6 6 6 6"/></svg></button>
        <input type="text" inputmode="numeric" class="pdf-page" aria-label="页码" value="1"><span class="pdf-total">/ –</span>
        <button type="button" class="icon-btn sm" data-pdf="next" title="下一页" aria-label="下一页"><svg viewBox="0 0 24 24"><path d="M9 6l6 6-6 6"/></svg></button>
      </div>
      <div class="pdf-zoom">
        <button type="button" class="icon-btn sm" data-pdf="out" title="缩小" aria-label="缩小"><svg viewBox="0 0 24 24"><path d="M6 12h12"/></svg></button>
        <button type="button" class="pdf-scale" data-pdf="fit" title="适合宽度">100%</button>
        <button type="button" class="icon-btn sm" data-pdf="in" title="放大" aria-label="放大"><svg viewBox="0 0 24 24"><path d="M6 12h12M12 6v12"/></svg></button>
      </div>
      <label class="search pdf-find">
        <svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4-4"/></svg>
        <input type="search" placeholder="在原文中查找（Enter 下一个）">
        <span class="pdf-count"></span>
      </label>`;
    const pageInput = els.tools.querySelector(".pdf-page");
    const scaleBtn = els.tools.querySelector(".pdf-scale");
    const findInput = els.tools.querySelector(".pdf-find input");
    const count = els.tools.querySelector(".pdf-count");

    const task = lib.getDocument({
      ...source,
      cMapUrl: abs(`${PDFJS}cmaps/`),
      cMapPacked: true,
      standardFontDataUrl: abs(`${PDFJS}standard_fonts/`),
      isEvalSupported: false,
      enableXfa: false,
    });
    ctx.cleanup.push(() => { task.destroy(); findController.setDocument?.(null); });
    const pdf = await task.promise;
    if (ctx.dead) return;
    const ready = new Promise((resolve) => eventBus.on("pagesinit", resolve, { once: true }));
    pdfViewer.setDocument(pdf);
    linkService.setDocument(pdf, null);
    await ready;
    if (ctx.dead) return;
    pdfViewer.currentScaleValue = "page-width";
    els.tools.querySelector(".pdf-total").textContent = `/ ${pdf.numPages}`;

    eventBus.on("pagechanging", ({ pageNumber }) => { pageInput.value = pageNumber; });
    eventBus.on("scalechanging", ({ scale }) => { scaleBtn.textContent = `${Math.round(scale * 100)}%`; });
    const showCount = ({ matchesCount, state }) => {
      if (!findInput.value.trim()) { count.textContent = ""; return; }
      if (state === 1) { count.textContent = "无结果"; return; } // FindState.NOT_FOUND
      if (matchesCount?.total) count.textContent = `${matchesCount.current}/${matchesCount.total}`;
    };
    eventBus.on("updatefindmatchescount", showCount);
    eventBus.on("updatefindcontrolstate", showCount);

    const find = (type, previous = false) => eventBus.dispatch("find", {
      source: null, type, query: findInput.value.trim(), caseSensitive: false, entireWord: false,
      highlightAll: true, findPrevious: previous, matchDiacritics: false, phraseSearch: true,
    });
    let findTimer = 0;
    findInput.addEventListener("input", () => { clearTimeout(findTimer); findTimer = setTimeout(() => find(""), 200); });
    findInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.isComposing) { e.preventDefault(); find("again", e.shiftKey); }
    });
    pageInput.addEventListener("change", () => {
      const n = Math.min(pdf.numPages, Math.max(1, parseInt(pageInput.value, 10) || 1));
      pdfViewer.currentPageNumber = n;
      pageInput.value = n;
    });
    pageInput.addEventListener("keydown", (e) => { if (e.key === "Enter") pageInput.dispatchEvent(new Event("change")); });
    els.tools.onclick = (e) => {
      const act = e.target.closest("[data-pdf]")?.dataset.pdf;
      if (act === "prev") pdfViewer.previousPage();
      else if (act === "next") pdfViewer.nextPage();
      else if (act === "fit") pdfViewer.currentScaleValue = "page-width";
      else if (act === "in") pdfViewer.currentScale = Math.min(5, pdfViewer.currentScale * 1.2);
      else if (act === "out") pdfViewer.currentScale = Math.max(0.25, pdfViewer.currentScale / 1.2);
    };

    // Cited text: jump to its page and mark it on the text layer once that page has one.
    let pending = null;
    eventBus.on("textlayerrendered", ({ pageNumber }) => {
      if (pending && pending.page === pageNumber) highlightOn(pending);
    });
    function highlightOn(target) {
      const div = pdfViewer.getPageView(target.page - 1)?.textLayer?.div;
      if (!div || !div.querySelector("span")) return;
      pending = null;
      container.querySelectorAll(".quote-hit").forEach((el) => el.classList.remove("quote-hit"));
      const first = markQuote(div, target.quote, { selector: "span" });
      if (first) requestAnimationFrame(() => first.scrollIntoView({ block: "center" }));
    }
    async function pageText(n) {
      const content = await (await pdf.getPage(n)).getTextContent();
      return norm(content.items.map((item) => item.str).join(""));
    }
    async function locateQuote(quote, pages) {
      // The cited page(s) first, then their neighbours; without a page, every page.
      const order = pages
        ? [...new Set([pages[0], ...range(pages[0], pages[1]), pages[0] - 1, pages[1] + 1])]
        : range(1, Math.min(pdf.numPages, PDF_SCAN_PAGES));
      let best = null;
      for (const n of order.filter((p) => p >= 1 && p <= pdf.numPages)) {
        if (ctx.dead) return null;
        const found = findQuote(await pageText(n), quote);
        if (found && (!best || found[1] - found[0] > best.length)) best = { page: n, length: found[1] - found[0] };
        if (best && best.length >= 40) break;
      }
      return best?.page || null;
    }
    ctx.focus = async ({ locator = "", quote = "" } = {}) => {
      const pages = pageRange(locator);
      if (pages) pdfViewer.currentPageNumber = Math.min(pdf.numPages, pages[0]);
      if (!norm(quote)) return;
      const page = await locateQuote(quote, pages);
      if (!page || ctx.dead) return;
      if (page !== pdfViewer.currentPageNumber) pdfViewer.currentPageNumber = page;
      pending = { page, quote };
      highlightOn(pending);
    };
    await ctx.focus(focus);
  }

  const range = (a, b) => Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);

  // ---------------------------------------------------------------- PowerPoint

  /** A deck: the PDF prepared in the background. A .pptx waits for it at most 1.5 s, then
      shows the slides drawn in the page (the next open gets the exact one); a legacy .ppt has
      no other rendering and waits for PowerPoint. */
  async function renderDeck(ctx, doc, focus) {
    loading();
    const quick = doc.preview.fallback === "slides";
    const res = await apiFetch(`${API}/documents/${encodeURIComponent(doc.doc_id)}/pdf${quick ? "?wait=1.5" : ""}`);
    if (ctx.dead) return;
    if (res.status === 202 || !res.ok) {
      if (quick) await renderSlides(ctx, doc, focus);
      else renderExternal(ctx, doc, res.ok || res.status === 202 ? "正在准备预览，稍后再打开。" : await errorDetail(res));
      return;
    }
    const data = new Uint8Array(await res.arrayBuffer());
    if (ctx.dead) return;
    await renderPdf(ctx, { data }, focus);
  }

  /** The slides from the deck's own XML: text, tables, charts' numbers and pictures in place.
      Fonts and backgrounds from the slide master are not drawn. */
  async function renderSlides(ctx, doc, focus) {
    loading();
    const deck = await (await fetchOk(`${API}/documents/${encodeURIComponent(doc.doc_id)}/slides`)).json();
    if (ctx.dead) return;
    setHow("没有用 PowerPoint 转换，版式是近似的");
    els.root.innerHTML = `<div class="orig-scroll orig-pad slides">${deck.slides.map((s) => slideHtml(s, deck)).join("")}</div>`;
    const scroller = els.root.firstElementChild;
    const urls = [];
    ctx.cleanup.push(() => urls.forEach((u) => URL.revokeObjectURL(u)));
    const fit = () => {
      const width = scroller.querySelector(".slide")?.clientWidth || 0;
      scroller.style.setProperty("--slide-scale", width && deck.width_pt ? String(width / deck.width_pt) : "1");
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(scroller);
    ctx.cleanup.push(() => observer.disconnect());
    // Pictures load as they scroll into view: a 50 MB deck should not fetch everything.
    const pictures = new IntersectionObserver((entries) => {
      entries.filter((e) => e.isIntersecting).forEach(async (entry) => {
        pictures.unobserve(entry.target);
        const res = await apiFetch(`${API}/documents/${encodeURIComponent(doc.doc_id)}/slides/media?part=${encodeURIComponent(entry.target.dataset.part)}`);
        if (!res.ok || ctx.dead) return;
        const url = URL.createObjectURL(await res.blob());
        urls.push(url);
        entry.target.src = url;
      });
    }, { root: scroller, rootMargin: "400px" });
    scroller.querySelectorAll("img[data-part]").forEach((img) => pictures.observe(img));
    ctx.cleanup.push(() => pictures.disconnect());
    ctx.focus = ({ locator = "", quote = "" } = {}) => {
      const pages = pageRange(locator);
      scroller.querySelectorAll(".quote-hit").forEach((el) => el.classList.remove("quote-hit"));
      const slide = pages ? scroller.querySelector(`[data-slide="${pages[0]}"]`) : null;
      const hit = norm(quote) ? markQuote(slide || scroller, quote) : null;
      (hit || slide)?.scrollIntoView({ block: hit ? "center" : "start" });
    };
    ctx.focus(focus);
  }

  function slideHtml(slide, deck) {
    const box = (s) => `left:${s.x * 100}%;top:${s.y * 100}%;width:${s.w * 100}%;height:${s.h * 100}%`;
    const shapes = slide.shapes.map((s) => {
      if (s.kind === "picture") return `<img class="sh" style="${box(s)}" data-part="${esc(s.image)}" alt="">`;
      if (s.kind === "table") {
        const rows = s.rows.map((r) => `<tr>${r.map((c) => `<td>${esc(c)}</td>`).join("")}</tr>`).join("");
        return `<div class="sh sh-table" style="${box(s)}"><table>${rows}</table></div>`;
      }
      if (s.kind === "chart" || s.kind === "diagram") {
        return `<div class="sh sh-lines" style="${box(s)}">${s.lines.map((l) => `<div>${esc(l)}</div>`).join("")}</div>`;
      }
      const body = s.paragraphs.map((p) => {
        const align = { ctr: "center", r: "right", just: "justify" }[p.align] || "left";
        const runs = p.runs.map((r) => {
          const style = [
            r.size ? `font-size:calc(${r.size}px * var(--slide-scale, 1))` : "",
            r.bold ? "font-weight:600" : "",
            /^[0-9a-f]{6}$/i.test(r.color) ? `color:#${r.color}` : "",
          ].filter(Boolean).join(";");
          return r.text === "\n" ? "<br>" : `<span style="${style}">${esc(r.text)}</span>`;
        }).join("");
        return `<p style="text-align:${align}">${runs || "&nbsp;"}</p>`;
      }).join("");
      return `<div class="sh sh-text" style="${box(s)}">${body}</div>`;
    }).join("");
    const notes = slide.notes ? `<div class="slide-notes">备注：${esc(slide.notes)}</div>` : "";
    const hidden = slide.hidden ? " · 隐藏" : "";
    return `<section class="slide-wrap" data-slide="${slide.number}">
      <div class="slide-no">第 ${slide.number} 张${hidden}</div>
      <div class="slide" style="aspect-ratio:${deck.aspect}">${shapes}</div>${notes}
    </section>`;
  }

  // ---------------------------------------------------------------- Word

  /** Word: the page-exact rendering prepared in the background. A .docx waits for it at
      most 1.5 s, then shows the in-page rendering at once (the next open gets the exact
      one); a legacy .doc has no other rendering and waits. */
  async function renderWord(ctx, doc, focus) {
    loading();
    const quick = doc.preview.fallback === "docx-preview";
    const res = await apiFetch(`${API}/documents/${encodeURIComponent(doc.doc_id)}/pdf${quick ? "?wait=1.5" : ""}`);
    if (ctx.dead) return;
    if (res.status === 202 || !res.ok) {
      if (quick) await renderDocx(ctx, doc, focus);
      else renderExternal(ctx, doc, "暂时无法在软件内显示这个文件。");
      return;
    }
    const data = new Uint8Array(await res.arrayBuffer());
    if (ctx.dead) return;
    await renderPdf(ctx, { data }, focus);
  }

  async function renderDocx(ctx, doc, focus) {
    loading();
    await loadScript(SCRIPTS.jszip);
    await loadScript(SCRIPTS.docx);
    const blob = await (await fetchOk(fileUrl(doc.doc_id))).blob();
    if (ctx.dead) return;
    els.root.innerHTML = '<div class="docx-scroll"><div class="docx-host"></div></div>';
    const host = els.root.querySelector(".docx-host");
    await window.docx.renderAsync(blob, host, undefined, {
      className: "docx",
      inWrapper: true,
      breakPages: true,
      ignoreLastRenderedPageBreak: true,
      renderHeaders: true,
      renderFooters: true,
      renderFootnotes: true,
      useBase64URL: false,
    });
    if (ctx.dead) return;
    const fit = () => {
      host.style.zoom = "";
      const page = host.querySelector("section.docx");
      const room = els.root.clientWidth - 24;
      if (page && page.offsetWidth > room) host.style.zoom = String(room / (page.offsetWidth + 40));
    };
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(els.root);
    ctx.cleanup.push(() => observer.disconnect());
    ctx.focus = ({ quote = "" } = {}) => {
      host.querySelectorAll(".quote-hit").forEach((el) => el.classList.remove("quote-hit"));
      const first = norm(quote) ? markQuote(host, quote) : null;
      if (first) first.scrollIntoView({ block: "center" });
    };
    ctx.focus(focus);
  }

  // ---------------------------------------------------------------- spreadsheets

  async function renderSheet(ctx, doc, focus) {
    loading();
    await loadScript(SCRIPTS.xlsx);
    const buffer = await (await fetchOk(fileUrl(doc.doc_id))).arrayBuffer();
    if (ctx.dead) return;
    const book = window.XLSX.read(buffer, { type: "array", cellDates: true, cellStyles: false });
    let active = 0;
    let marked = null;

    const draw = () => {
      els.tools.innerHTML = `<div class="sheet-tabs" role="tablist">${book.SheetNames.map((name, i) =>
        `<button type="button" class="tab${i === active ? " active" : ""}" data-sheet="${i}" role="tab" title="${esc(name)}">${esc(name)}</button>`).join("")}</div>`;
      els.root.innerHTML = `<div class="orig-scroll orig-pad">${sheetHtml(book.Sheets[book.SheetNames[active]], marked)}</div>`;
    };
    els.tools.onclick = (e) => {
      const tab = e.target.closest("[data-sheet]");
      if (!tab) return;
      active = Number(tab.dataset.sheet);
      marked = null;
      draw();
    };
    ctx.focus = ({ locator = "" } = {}) => {
      const sheet = locator.match(/sheet:([^;]+)/);
      const rows = locator.match(/rows:(\d+)-(\d+)/);
      if (sheet) {
        const want = sheet[1].trim();
        const index = book.SheetNames.findIndex((name) => name.trim() === want);
        if (index >= 0) active = index;
      }
      marked = rows ? [Number(rows[1]), Number(rows[2])] : null;
      draw();
      const row = els.root.querySelector("tr.row-hit");
      if (row) requestAnimationFrame(() => row.scrollIntoView({ block: "center" }));
    };
    ctx.focus(focus);
    ctx.cleanup.push(() => { els.tools.onclick = null; });
  }

  /** A worksheet as a table with Excel's column letters and row numbers; merged cells span. */
  function sheetHtml(ws, marked) {
    const X = window.XLSX;
    if (!ws || !ws["!ref"]) return '<div class="viewer-empty">这个工作表是空的。</div>';
    const full = X.utils.decode_range(ws["!ref"]);
    const last = { r: Math.min(full.e.r, full.s.r + SHEET_MAX_ROWS - 1), c: Math.min(full.e.c, full.s.c + SHEET_MAX_COLS - 1) };
    const covered = new Set();
    const spans = new Map();
    for (const m of ws["!merges"] || []) {
      spans.set(`${m.s.r},${m.s.c}`, [m.e.r - m.s.r + 1, m.e.c - m.s.c + 1]);
      for (let r = m.s.r; r <= m.e.r; r++) for (let c = m.s.c; c <= m.e.c; c++) if (r !== m.s.r || c !== m.s.c) covered.add(`${r},${c}`);
    }
    let html = '<table class="xls"><thead><tr><th class="corner"></th>';
    for (let c = full.s.c; c <= last.c; c++) html += `<th>${X.utils.encode_col(c)}</th>`;
    html += "</tr></thead><tbody>";
    for (let r = full.s.r; r <= last.r; r++) {
      const excelRow = r + 1;
      const hit = marked && excelRow >= marked[0] && excelRow <= marked[1];
      html += `<tr${hit ? ' class="row-hit"' : ""}><th>${excelRow}</th>`;
      for (let c = full.s.c; c <= last.c; c++) {
        if (covered.has(`${r},${c}`)) continue;
        const cell = ws[X.utils.encode_cell({ r, c })];
        const span = spans.get(`${r},${c}`);
        const attrs = span ? `${span[0] > 1 ? ` rowspan="${span[0]}"` : ""}${span[1] > 1 ? ` colspan="${span[1]}"` : ""}` : "";
        const text = cell ? (cell.w ?? X.utils.format_cell(cell)) : "";
        const num = cell && cell.t === "n" ? ' class="num"' : "";
        html += `<td${attrs}${num}>${esc(text)}</td>`;
      }
      html += "</tr>";
    }
    html += "</tbody></table>";
    const cut = [];
    if (last.r < full.e.r) cut.push(`只显示前 ${SHEET_MAX_ROWS} 行（共 ${full.e.r - full.s.r + 1} 行）`);
    if (last.c < full.e.c) cut.push(`前 ${SHEET_MAX_COLS} 列`);
    return html + (cut.length ? `<p class="viewer-note">${cut.join("，")}，完整内容请用 Excel 打开。</p>` : "");
  }

  // ---------------------------------------------------------------- text, markdown, images

  async function readText(docId) {
    const bytes = new Uint8Array(await (await fetchOk(fileUrl(docId))).arrayBuffer());
    try {
      return new TextDecoder("utf-8", { fatal: true }).decode(bytes).replace(/^﻿/, "");
    } catch (_) {
      return new TextDecoder("gbk").decode(bytes); // older Chinese text files
    }
  }

  function mdInline(text) {
    return esc(text)
      .replace(/!\[([^\]]*)\]\([^)]*\)/g, (_, alt) => `<span class="md-img">〔图片${alt ? `：${alt}` : ""}〕</span>`)
      .replace(/\[([^\]]+)\]\(([^)\s]+)[^)]*\)/g, (_, label, href) => `<span class="md-link" title="${href}">${label}</span>`)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>");
  }

  /** Markdown as a document: headings by level, lists, quotes, code, pipe tables. */
  function markdownHtml(src) {
    const lines = String(src).replace(/\r/g, "").split("\n");
    const out = [];
    let para = [];
    const flush = () => { if (para.length) out.push(`<p>${para.map(mdInline).join("<br>")}</p>`); para = []; };
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      let m;
      if (/^\s*```/.test(line)) {
        flush();
        const code = [];
        while (++i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i]);
        out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
      } else if (!line.trim()) {
        flush();
      } else if ((m = line.match(/^\s*(#{1,6})\s+(.*?)\s*#*\s*$/))) {
        flush();
        out.push(`<h${m[1].length}>${mdInline(m[2])}</h${m[1].length}>`);
      } else if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
        flush();
        out.push("<hr>");
      } else if (/^\s*\|.*\|\s*$/.test(line) && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1] || "")) {
        flush();
        const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        let html = `<table><thead><tr>${cells(line).map((c) => `<th>${mdInline(c)}</th>`).join("")}</tr></thead><tbody>`;
        i++;
        while (i + 1 < lines.length && /^\s*\|.*\|\s*$/.test(lines[i + 1])) html += `<tr>${cells(lines[++i]).map((c) => `<td>${mdInline(c)}</td>`).join("")}</tr>`;
        out.push(`${html}</tbody></table>`);
      } else if (/^\s*>/.test(line)) {
        flush();
        const quote = [line.replace(/^\s*>\s?/, "")];
        while (i + 1 < lines.length && /^\s*>/.test(lines[i + 1])) quote.push(lines[++i].replace(/^\s*>\s?/, ""));
        out.push(`<blockquote>${quote.map(mdInline).join("<br>")}</blockquote>`);
      } else if ((m = line.match(/^(\s*)([-*+•]|\d+[.、)])\s+(.*)$/))) {
        flush();
        const ordered = /\d/.test(m[2]);
        const items = [m[3]];
        while (i + 1 < lines.length && (m = lines[i + 1].match(/^(\s*)([-*+•]|\d+[.、)])\s+(.*)$/)) && /\d/.test(m[2]) === ordered) { items.push(m[3]); i++; }
        const tag = ordered ? "ol" : "ul";
        out.push(`<${tag}>${items.map((item) => `<li>${mdInline(item)}</li>`).join("")}</${tag}>`);
      } else {
        para.push(line);
      }
    }
    flush();
    return out.join("");
  }

  async function renderTextual(ctx, doc, focus, kind) {
    loading();
    const text = await readText(doc.doc_id);
    if (ctx.dead) return;
    els.root.innerHTML = kind === "markdown"
      ? `<div class="orig-scroll orig-pad"><article class="md-doc">${markdownHtml(text)}</article></div>`
      : `<div class="orig-scroll orig-pad"><pre class="txt-doc">${esc(text)}</pre></div>`;
    const host = els.root.firstElementChild;
    ctx.focus = ({ quote = "" } = {}) => {
      host.querySelectorAll(".quote-hit").forEach((el) => el.classList.remove("quote-hit"));
      // A plain-text file is one <pre>: mark that line range by wrapping the match.
      const first = norm(quote) ? (kind === "markdown" ? markQuote(host, quote) : markPre(host.querySelector("pre"), quote)) : null;
      if (first) requestAnimationFrame(() => first.scrollIntoView({ block: "center" }));
    };
    ctx.focus(focus);
  }

  /** In one big text node, wrap the quoted stretch in <mark>. */
  function markPre(pre, quote) {
    const text = pre.textContent;
    const positions = [];
    let flat = "";
    for (let i = 0; i < text.length; i++) {
      const piece = norm(text[i]);
      for (const ch of piece) { flat += ch; positions.push(i); }
    }
    const found = findQuote(flat, quote);
    if (!found) return null;
    const start = positions[found[0]];
    const end = positions[found[1] - 1] + 1;
    pre.innerHTML = `${esc(text.slice(0, start))}<mark class="quote-hit">${esc(text.slice(start, end))}</mark>${esc(text.slice(end))}`;
    return pre.querySelector("mark");
  }

  async function renderImage(ctx, doc) {
    loading();
    const blob = await (await fetchOk(imageUrl(doc.doc_id))).blob();
    if (ctx.dead) return;
    const url = URL.createObjectURL(blob);
    ctx.cleanup.push(() => URL.revokeObjectURL(url));
    els.root.innerHTML = `<div class="orig-scroll img-doc"><img src="${url}" alt="${esc(doc.source)}"></div>`;
  }

  function renderExternal(ctx, doc, why) {
    const ext = (doc.source.split(".").pop() || "").toUpperCase();
    els.root.innerHTML = `<div class="orig-empty">
      <p>${esc(why || `暂不支持在软件内预览 ${ext} 文件。`)}</p>
      <div class="orig-empty-actions">
        <button type="button" class="btn primary" data-orig="open">用系统默认程序打开</button>
        <button type="button" class="btn ghost" data-orig="parsed">查看解析文本</button>
      </div>
    </div>`;
  }

  // ---------------------------------------------------------------- public

  async function open(doc, focus = {}) {
    close();
    const ctx = { docId: doc.doc_id, cleanup: [], focus: () => {}, dead: false };
    current = ctx;
    els.tools.innerHTML = "";
    setHow("");
    const plan = doc.preview || { renderer: "external" };
    try {
      if (plan.renderer === "pdfjs") {
        loading();
        await renderPdf(ctx, { url: fileUrl(doc.doc_id), httpHeaders: authHeaders(), withCredentials: false }, focus);
      } else if (plan.renderer === "word-pdf") await renderWord(ctx, doc, focus);
      else if (plan.renderer === "docx-preview") await renderDocx(ctx, doc, focus);
      else if (plan.renderer === "slides-pdf") await renderDeck(ctx, doc, focus);
      else if (plan.renderer === "slides") await renderSlides(ctx, doc, focus);
      else if (plan.renderer === "sheetjs") await renderSheet(ctx, doc, focus);
      else if (plan.renderer === "markdown" || plan.renderer === "text") await renderTextual(ctx, doc, focus, plan.renderer);
      else if (plan.renderer === "image") await renderImage(ctx, doc);
      else renderExternal(ctx, doc);
    } catch (err) {
      if (ctx.dead) return;
      console.error(err);
      els.tools.innerHTML = "";
      renderExternal(ctx, doc, `无法在软件内显示原文：${err.message || err}`);
    }
  }

  /** Jump to another place in the document already shown. */
  function focus(docId, target) {
    if (current && current.docId === docId) return Promise.resolve(current.focus(target || {}));
    return null;
  }

  function close() {
    if (!current) return;
    current.dead = true;
    for (const fn of current.cleanup.splice(0)) {
      try { fn(); } catch (_) { /* already gone */ }
    }
    current = null;
    els.root.innerHTML = "";
    els.tools.innerHTML = "";
    els.tools.onclick = null;
    setHow("");
  }

  const showing = (docId) => current?.docId === docId;

  return { open, focus, close, showing, findQuote, norm };
})();
