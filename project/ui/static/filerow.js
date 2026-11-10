"use strict";

/* Compact file rows for "find file" results, shared by the chat (app.js) and the
   quick-search window (quick.js): icon, name, last two folders, modified time; actions
   on hover; one grey line saying why it matched, with the snippets behind a toggle.
   Files with the same name in several places stack into one row that fans out. */
(function (root) {
  const KIND = {
    ".pdf": ["PDF", "pdf"], ".docx": ["DOC", "word"], ".doc": ["DOC", "word"],
    ".xlsx": ["XLS", "excel"], ".xls": ["XLS", "excel"], ".csv": ["CSV", "excel"],
    ".pptx": ["PPT", "ppt"], ".ppt": ["PPT", "ppt"], ".md": ["MD", "text"], ".markdown": ["MD", "text"],
    ".txt": ["TXT", "text"], ".png": ["IMG", "image"], ".jpg": ["IMG", "image"], ".jpeg": ["IMG", "image"],
  };
  const COLLAPSED = 5;
  const LOST = new Set(["missing", "offline"]);
  const svg = (name) => (root.Icons ? root.Icons.svg(name) : "");

  function esc(text) {
    return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }

  function icon(ext) {
    const key = (ext || "").toLowerCase();
    const [label, kind] = KIND[key] || [key.replace(".", "").slice(0, 4).toUpperCase() || "FILE", "other"];
    return `<span class="file-icon kind-${kind}">${esc(label)}</span>`;
  }

  /** "…\\项目\\2026" for a file in C:\a\b\项目\2026\x.pdf. */
  function folderTail(path) {
    const parts = String(path || "").split(/[\\/]+/).filter(Boolean);
    parts.pop();
    if (!parts.length) return "";
    const tail = parts.slice(-2).join("\\");
    return parts.length > 2 ? `…\\${tail}` : tail;
  }

  function locatorLabel(locator) {
    if (!locator) return "";
    const page = locator.match(/^page:(\d+)(?:-(\d+))?$/);
    if (page) return page[2] && page[2] !== page[1] ? `第 ${page[1]}–${page[2]} 页` : `第 ${page[1]} 页`;
    const slide = locator.match(/^slide:(\d+)(?:-(\d+))?$/);
    if (slide) return slide[2] && slide[2] !== slide[1] ? `第 ${slide[1]}–${slide[2]} 张` : `第 ${slide[1]} 张`;
    const sheet = locator.match(/sheet:([^;]+)/);
    const rows = locator.match(/rows:(\d+)-(\d+)/);
    const rowText = rows ? (rows[1] === rows[2] ? `第 ${rows[1]} 行` : `第 ${rows[1]}–${rows[2]} 行`) : "";
    if (sheet) return [sheet[1].trim(), rowText].filter(Boolean).join(" · ");
    if (rowText) return rowText;
    const table = locator.match(/^table:(\d+)$/);
    if (table) return `表格 ${table[1]}`;
    return "";
  }

  function shorten(text, limit) {
    const clean = String(text || "").replace(/\s+/g, " ").replace(/…$/, "").trim();
    return clean.length > limit ? `${clean.slice(0, limit)}…` : clean;
  }

  function fmtDate(iso) {
    if (!iso) return "";
    const date = iso.slice(0, 10);
    const today = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const local = `${today.getFullYear()}-${pad(today.getMonth() + 1)}-${pad(today.getDate())}`;
    if (date === local) return `今天 ${iso.slice(11, 16)}`;
    return date.slice(0, 4) === local.slice(0, 4) ? `${Number(date.slice(5, 7))}月${Number(date.slice(8, 10))}日` : date;
  }

  /** One short line: what in the file matched, and where. */
  function matchReason(f, filters) {
    const snip = (f.snippets || [])[0];
    const by = f.matched_by || [];
    const where = locatorLabel(snip?.locator || "");
    let what = "";
    if (by.includes("type")) {
      // A document of the type asked for ("找一下我的简历"): what kind it is says why.
      return `类型为${f.type_label || f.doc_type || filters?.doc_type || ""}`;
    }
    if (by.includes("mention") && filters?.doc_type_word) {
      const own = f.doc_type ? `${f.type_label || f.doc_type} · ` : "";
      return [`${own}提到“${filters.doc_type_word}”`, where].filter(Boolean).join(" · ");
    }
    if (by.includes("content") || by.includes("summary")) {
      const heading = (snip?.heading || "").split(/\s*>\s*/).pop();
      what = heading && heading.length <= 28 ? heading : shorten(filters?.topic, 20) || shorten(snip?.text, 18) || "内容";
    } else if (by.includes("name")) {
      what = "文件名";
    } else if (by.includes("metadata")) {
      what = [filters?.time_phrase, ...(filters?.file_types || [])].filter(Boolean).join(" · ") || "筛选条件";
    }
    return [what, where].filter(Boolean).join(" · ");
  }

  /** Indexes of the files grouped by name (same name, different places), in first-seen order. */
  function groupByName(files) {
    const groups = [];
    const at = new Map();
    files.forEach((f, i) => {
      const key = String(f.name || "").toLowerCase();
      if (key && at.has(key)) groups[at.get(key)].push(i);
      else {
        if (key) at.set(key, groups.length);
        groups.push([i]);
      }
    });
    return groups;
  }

  function row(f, i, { filters, active = false, actions = true, tag = "li", count = 0 } = {}) {
    const lost = LOST.has(f.status);
    const reason = matchReason(f, filters);
    const snippets = (f.snippets || []).filter((s) => s.text);
    const time = f.mtime ? `修改于 ${f.mtime.slice(0, 16).replace("T", " ")}` : "";
    return `<${tag} class="frow${lost ? " lost" : ""}${active ? " active" : ""}" data-i="${i}" data-file="${esc(f.file_id)}"
        data-name="${esc(f.name)}" data-locator="${esc(snippets[0]?.locator || "")}" title="${esc(f.path || f.name)}">
      <div class="frow-main">
        ${icon(f.ext)}
        <span class="frow-name">${esc(f.name)}</span>
        <span class="frow-folder">${esc(folderTail(f.path))}</span>
        ${count ? `<button type="button" class="frow-count" data-act="versions" title="同名文件在 ${count} 个位置">${count} 个位置${svg("chevron")}</button>` : ""}
        <span class="frow-time" title="${esc(time)}">${lost ? `<em>${f.status === "offline" ? "磁盘未连接" : "原文件已丢失"}</em>` : esc(fmtDate(f.mtime))}</span>
        ${actions ? `<span class="frow-actions">
          <button type="button" data-act="read" title="在阅读面板打开" aria-label="打开">${svg("doc")}</button>
          <button type="button" data-act="reveal" title="在资源管理器中显示" aria-label="在资源管理器中显示"${lost ? " disabled" : ""}>${svg("folder")}</button>
          <button type="button" data-act="ask" title="只问这个文件" aria-label="只问这个文件">${svg("chat")}</button>
        </span>` : ""}
      </div>
      ${reason || snippets.length ? `<div class="frow-why">
        ${reason ? `<span>匹配：${esc(reason)}</span>` : ""}
        ${snippets.length ? `<button type="button" class="frow-toggle" data-act="snips">${svg("chevron")}片段</button>` : ""}
      </div>` : ""}
      ${snippets.length ? `<div class="frow-snips" hidden>${snippets.map((s) => `<blockquote data-locator="${esc(s.locator || "")}">${esc(shorten(s.text, 180))}${
        locatorLabel(s.locator) ? ` <em>${esc(locatorLabel(s.locator))}</em>` : ""}</blockquote>`).join("")}</div>` : ""}
    </${tag}>`;
  }

  /** The list; past COLLAPSED entries the rest wait behind "显示全部 N 个". A stack holding the
      active file opens, so keyboard selection never lands on a hidden row. */
  function list(files, { filters, active = -1, expanded = false, actions = true } = {}) {
    const groups = groupByName(files);
    const fold = !expanded && groups.length > COLLAPSED;
    const rows = groups.map((g, gi) => {
      const opts = { filters, actions };
      let html;
      if (g.length === 1) {
        html = row(files[g[0]], g[0], { ...opts, active: g[0] === active });
      } else {
        const open = g.slice(1).includes(active);
        html = `<li class="frow-stack${open ? " open" : ""}">${row(files[g[0]], g[0], { ...opts, active: g[0] === active, tag: "div", count: g.length })}
          <ul class="frow-versions"${open ? "" : " hidden"}>${g.slice(1).map((i) => row(files[i], i, { ...opts, active: i === active })).join("")}</ul></li>`;
      }
      return fold && gi >= COLLAPSED && !g.includes(active) ? html.replace(/^<li /, "<li hidden ") : html;
    });
    return `<ul class="frows">${rows.join("")}</ul>${
      fold ? `<button type="button" class="frows-more" data-act="more">显示全部 ${files.length} 个</button>` : ""}`;
  }

  /** Unfold a list folded by list(). */
  function expand(more) {
    more.previousElementSibling?.querySelectorAll(":scope > [hidden]").forEach((li) => { li.hidden = false; });
    more.remove();
  }

  /** What a click on a list means: {act, index, row, locator?} or null. */
  function hit(event) {
    const more = event.target.closest(".frows-more");
    if (more) { expand(more); return { act: "more" }; }
    const li = event.target.closest(".frow");
    if (!li) return null;
    const index = Number(li.dataset.i);
    const button = event.target.closest("[data-act]");
    if (button?.disabled) return { act: "none", index, row: li };
    if (button?.dataset.act === "versions") {
      const stack = li.closest(".frow-stack");
      const open = stack.classList.toggle("open");
      stack.querySelector(".frow-versions").hidden = !open;
      return { act: "none", index, row: li };
    }
    if (button?.dataset.act === "snips") {
      const box = li.querySelector(".frow-snips");
      box.hidden = !box.hidden;
      button.classList.toggle("open", !box.hidden);
      return { act: "none", index, row: li };
    }
    const quote = event.target.closest(".frow-snips blockquote");
    if (quote) return { act: "read", index, row: li, locator: quote.dataset.locator, quote: quote.firstChild?.textContent || "" };
    return { act: button?.dataset.act || "read", index, row: li };
  }

  root.FileRows = { list, hit, expand, icon, folderTail, locatorLabel, matchReason, groupByName, COLLAPSED };
})(typeof window !== "undefined" ? window : globalThis);
