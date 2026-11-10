"use strict";

/* Quick-search window: the desktop shell opens it with a global shortcut. Every keystroke
   (debounced) asks /files/search, which uses rules only (no LLM), so it stays fast. */
const DESKTOP = window.__RAG_DESKTOP__ || null;
const TAURI = window.__TAURI__ || null;
const API = `${DESKTOP?.apiBase || ""}/api/v1`;
const $ = (sel) => document.querySelector(sel);
const input = $("#q");
const list = $("#list");
const empty = $("#empty");
$("#pet").outerHTML = Icons.mascot();
const pet = document.querySelector(".quick-head .mascot");
const state = { files: [], filters: null, active: 0, expanded: false, timer: 0, seq: 0 };

async function apiFetch(path, options = {}) {
  const headers = new Headers(options.headers || {});
  if (DESKTOP?.token) headers.set("X-App-Token", DESKTOP.token);
  try {
    return await fetch(`${API}${path}`, { ...options, headers });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    // fetch() only says "Failed to fetch" when nothing answers.
    throw new Error("无法连接后台服务，程序可能已退出或正在重启");
  }
}

function render() {
  empty.hidden = state.files.length > 0;
  list.hidden = !state.files.length;
  // Moving the selection past the folded rows unfolds the list.
  if (state.active >= FileRows.COLLAPSED) state.expanded = true;
  list.innerHTML = FileRows.list(state.files, { filters: state.filters, active: state.active, expanded: state.expanded });
  list.querySelector(".frow.active")?.scrollIntoView({ block: "nearest" });
}

async function search() {
  const q = input.value.trim();
  const seq = ++state.seq;
  if (!q) { state.files = []; empty.textContent = "输入关键词，按文件名和内容找"; render(); return; }
  $("#quick").classList.add("loading");
  pet.classList.add("working");
  try {
    const res = await apiFetch(`/files/search?q=${encodeURIComponent(q)}&limit=8`);
    const data = await res.json().catch(() => ({}));
    if (seq !== state.seq) return; // a newer query is on its way
    if (!res.ok) throw new Error((Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("；") : data.detail) || `服务返回 ${res.status}`);
    state.files = data.files || [];
    state.filters = data.filters || null;
    prepare(state.files.slice(0, 3));
    state.active = 0;
    state.expanded = false;
    empty.textContent = `${Icons.NAME}翻遍了，没有找到符合条件的文件`;
  } catch (err) {
    if (seq !== state.seq) return;
    state.files = [];
    empty.textContent = `搜索失败：${err.message || err}`;
  } finally {
    if (seq === state.seq) {
      $("#quick").classList.remove("loading");
      pet.classList.remove("working");
    }
  }
  render();
}

/** The top Word results get their preview prepared in the background, so opening one is quick. */
function prepare(files) {
  const ids = files.filter((f) => /\.(docx|doc)$/i.test(f.name || "")).map((f) => f.file_id);
  if (!ids.length) return;
  apiFetch("/documents/prepare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc_ids: ids }),
  }).catch(() => {});
}

function hide() {
  if (TAURI?.core) TAURI.core.invoke("hide_quick").catch(() => {});
}

async function openReader(f, locator, quote) {
  if (!TAURI?.core) return;
  await TAURI.core.invoke("open_reader", {
    fileId: f.file_id,
    locator: locator ?? f.snippets?.[0]?.locator ?? "",
    quote: quote ?? f.snippets?.[0]?.text ?? "",
  });
}

/** Default action (click, Enter): the original in the reader panel. Ctrl: the default program. */
async function open(f, external) {
  if (!f) return;
  if (!external || f.status !== "indexed") { await openReader(f); return; }
  const res = await apiFetch(`/files/${encodeURIComponent(f.file_id)}/open`, { method: "POST" });
  if (res.ok) hide();
  else await openReader(f); // the original moved away: the indexed text is still readable
}

async function reveal(f) {
  const res = await apiFetch(`/files/${encodeURIComponent(f.file_id)}/reveal`, { method: "POST" });
  if (res.ok) hide();
}

async function ask(f) {
  if (TAURI?.core) await TAURI.core.invoke("ask_about_file", { fileId: f.file_id, name: f.name });
}

input.addEventListener("input", () => {
  clearTimeout(state.timer);
  state.timer = setTimeout(search, 220);
});
input.addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (!state.files.length) return;
    state.active = (state.active + (e.key === "ArrowDown" ? 1 : -1) + state.files.length) % state.files.length;
    render();
  } else if (e.key === "Enter" && !e.isComposing) {
    e.preventDefault();
    open(state.files[state.active], e.ctrlKey);
  }
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { e.preventDefault(); hide(); }
});
list.addEventListener("click", (e) => {
  const hit = FileRows.hit(e);
  if (!hit || hit.act === "none" || hit.act === "more") {
    if (hit?.act === "more") state.expanded = true;
    input.focus();
    return;
  }
  const f = state.files[hit.index];
  if (hit.act === "reveal") reveal(f);
  else if (hit.act === "ask") ask(f);
  else if (hit.locator !== undefined) openReader(f, hit.locator, hit.quote);
  else open(f, e.ctrlKey);
});

try {
  const saved = localStorage.getItem("theme");
  if (saved) document.documentElement.dataset.theme = saved;
} catch (_) {}

// Each time the shortcut shows the window: start from an empty, focused box.
TAURI?.event?.listen("quick-show", () => {
  input.value = "";
  state.files = [];
  state.expanded = false;
  state.seq++;
  render();
  input.focus();
});
render();
input.focus();
