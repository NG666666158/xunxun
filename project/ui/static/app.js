"use strict";

/* The desktop shell (Tauri) serves this page itself and injects where the backend runs
   plus a per-launch token; as a plain web page everything is same-origin and open. */
const DESKTOP = window.__RAG_DESKTOP__ || null;
const TAURI = window.__TAURI__ || null;
const API = `${DESKTOP?.apiBase || ""}/api/v1`;

/** fetch() rejects with a bare "Failed to fetch" when nothing answers; say what that means. */
const BACKEND_UNREACHABLE = "无法连接后台服务：程序可能已退出或正在重启。请稍后重试；如果一直出现，请重新打开应用，或查看数据目录下 logs 里的 backend.log。";

/** fetch() that carries the desktop token on every backend request. */
async function apiFetch(url, options = {}) {
  const headers = new Headers(options.headers || {});
  if (DESKTOP?.token) headers.set("X-App-Token", DESKTOP.token);
  try {
    return await fetch(url, { ...options, headers });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new Error(BACKEND_UNREACHABLE);
  }
}

/** The backend's reason for a failed response ({"detail": ...}), or its status. */
async function errorDetail(res) {
  const data = await res.json().catch(() => ({}));
  const detail = Array.isArray(data.detail) ? data.detail.map((d) => d.msg).join("；") : data.detail;
  return detail || `服务返回 ${res.status}`;
}

const $ = (sel) => document.querySelector(sel);
const els = {
  app: $("#app"),
  thread: $("#thread"),
  empty: $("#empty"),
  scroller: $("#scroller"),
  composer: $("#composer"),
  input: $("#input"),
  send: $("#send"),
  history: $("#historyList"),
  docList: $("#docList"),
  docCount: $("#docCount"),
  uploads: $("#uploads"),
  dropzone: $("#dropzone"),
  fileInput: $("#fileInput"),
  status: $("#status"),
  statusText: $("#statusText"),
  toast: $("#toast"),
  viewTitle: $("#viewTitle"),
};

const PET = Icons.NAME; // 寻寻, the puppy that fetches answers from the user's files
const ICONS = {
  check: Icons.svg("check"),
  alert: Icons.svg("alert"),
  search: Icons.svg("search"),
  rewrite: Icons.svg("compose"),
  clock: Icons.svg("clock"),
  chev: Icons.svg("chevron", "chev"),
  trash: Icons.svg("trash"),
  x: Icons.svg("x"),
  folder: Icons.svg("folder"),
  locate: Icons.svg("scope"),
  reveal: Icons.svg("folderOpen"),
  refresh: Icons.svg("refresh"),
  shield: Icons.svg("shield"),
  copy: Icons.svg("copy"),
  doc: Icons.svg("doc"),
  chat: Icons.svg("chat"),
  scope: Icons.svg("scope"),
  followup: Icons.svg("followup"),
  more: Icons.svg("more"),
};

/** Put the sprite icons named by data-icon="…" into static markup. */
function hydrateIcons(scope = document) {
  scope.querySelectorAll("[data-icon]").forEach((el) => {
    const svg = Icons.svg(el.dataset.icon, el.classList.contains("chev") ? "chev" : "");
    if (el.tagName === "SPAN" && (!el.className || el.className === "chev")) el.outerHTML = svg;
    else { el.insertAdjacentHTML("afterbegin", svg); el.removeAttribute("data-icon"); }
  });
}

const STATUS_LABEL = {
  pending: "排队中",
  indexing: "入库中",
  indexed: "已入库",
  named: "只按文件名",
  failed: "入库失败",
  unsupported: "不支持",
  missing: "原文件已丢失",
  offline: "所在磁盘离线",
  recycled: "在回收站",
};
const LOST = new Set(["missing", "offline", "recycled"]);
const IMAGE_MODE_LABEL = { local: "本机识别", cloud: "云端识别", name: "只记文件名" };

const INTENT_LABEL = {
  knowledge: "文档问答",
  find_file: "找文件",
  chitchat: "闲聊",
  clarification: "需要澄清",
};

/* ---------------- utilities ---------------- */

function esc(text) {
  return String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function toast(message) {
  if (!message) return;
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => els.toast.classList.remove("show"), 2600);
}

function uid() {
  if (crypto.randomUUID) return crypto.randomUUID();
  return "t-" + Date.now().toString(36) + Math.random().toString(36).slice(2);
}

/* ---------------- conversation history (in the data folder, see convstore.js) ---------------- */

const conversationSaver = new ConvStore.Saver({
  put: async (conv, closing) => {
    const { id, ...rest } = conv;
    const body = JSON.stringify(rest);
    // Only a small request can outlive the page.
    await api(`/conversations/${encodeURIComponent(id)}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body,
      keepalive: closing && body.length < 60000,
    });
  },
  remove: (id, closing) => api(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE", keepalive: closing }),
  onFailing: (failing, err) => {
    state.saveIssue = failing ? err?.message || "保存失败" : "";
    renderHistory();
  },
});
window.addEventListener("pagehide", () => conversationSaver.flush());

function saveStore(conv = currentConversation()) {
  if (conv && state.conversations.includes(conv)) conversationSaver.save(conv);
}

/** The sidebar's list, after moving anything localStorage still holds; messages are read on opening. */
async function loadConversations() {
  state.historyState = "loading";
  renderHistory();
  try {
    let moved;
    try {
      moved = await ConvStore.moveLegacy(localStorage, (conversations) => postJson("/conversations/import", { conversations }), uid);
    } catch (err) {
      // localStorage keeps them and the next launch tries again; until then they aren't listed.
      toast(`旧对话没能迁移：${err.message || err}`);
    }
    if (moved?.unreadable) toast("有一份旧对话记录读不出来，已原样保留");
    const data = await api("/conversations");
    state.conversations = ConvStore.mergeList(state.conversations, data.conversations || []);
    state.historyState = "ready";
  } catch (err) {
    state.historyState = "failed";
    state.historyError = err.message || String(err);
  }
  renderHistory();
}

async function loadMessages(conv) {
  if (!conv.loading) {
    conv.loading = api(`/conversations/${encodeURIComponent(conv.id)}`).finally(() => { conv.loading = null; });
  }
  const data = await conv.loading;
  if (!conv.messages) {
    conv.messages = ConvStore.settleMessages(data.messages || [], uid);
    conv.scope = data.scope || null;
  }
}

/** Show a conversation from the history; a click on another one meanwhile wins. */
async function openConversation(id) {
  const conv = state.conversations.find((c) => c.id === id);
  if (!conv) return;
  state.opening = id;
  if (!conv.messages) {
    try {
      await loadMessages(conv);
    } catch (err) {
      if (state.opening === id) toast(`打不开这段对话：${err.message || err}`);
      return;
    }
    if (state.opening !== id) return;
  }
  state.opening = null;
  state.current = id;
  renderConversation();
  showView("chat");
  followupsForOpened(conv);
}

function scrollToBottom(force) {
  const s = els.scroller;
  const near = s.scrollHeight - s.scrollTop - s.clientHeight < 160;
  if (force || near) s.scrollTop = s.scrollHeight;
}

/* ---------------- answer rendering ---------------- */

/** Turn inline "[来源：X｜位置：Y]" markers into numbered references (citations.js). */
function linkCitations(answer, citations) {
  const { text, marks, sources } = Citations.link(answer, citations);
  let html = renderMarkdown(text);
  // Collapse adjacent duplicate markers (e.g. same source cited twice in a row).
  html = html.replace(/\u0000(\d+)\u0000/g, (_, t) => {
    const n = marks[Number(t)];
    return `<span class="cite" data-n="${n}" title="${esc(sources[n - 1].source)}">${n}</span>`;
  });
  html = html.replace(/(<span class="cite" data-n="(\d+)"[^>]*>\d+<\/span>)(\s*<span class="cite" data-n="\2"[^>]*>\d+<\/span>)+/g, "$1");
  return { html, sources };
}

function inline(text) {
  return esc(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
}

function renderMarkdown(src) {
  const lines = String(src).replace(/\r/g, "").split("\n");
  const out = [];
  let para = [];
  let list = null;

  const flushPara = () => {
    if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`);
    para = [];
  };
  const flushList = () => {
    if (list) out.push(`<${list.type}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.type}>`);
    list = null;
  };

  let code = null;

  for (const raw of lines) {
    const line = raw.trimEnd();
    let m;
    // Fenced code keeps its lines verbatim; an unclosed fence (mid-stream) still renders.
    if (code) {
      if (/^\s*```/.test(line)) { out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`); code = null; }
      else code.push(raw);
      continue;
    }
    if (/^\s*```/.test(line)) { flushPara(); flushList(); code = []; continue; }
    if (!line.trim()) { flushPara(); flushList(); continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flushPara(); flushList(); out.push("<hr>"); continue; }
    if ((m = line.match(/^\s*#{1,6}\s+(.*)$/))) {
      flushPara(); flushList();
      out.push(`<h4>${inline(m[1])}</h4>`);
    } else if ((m = line.match(/^\s*(\d+)[.、)]\s+(.*)$/))) {
      flushPara();
      if (!list || list.type !== "ol") { flushList(); list = { type: "ol", items: [] }; }
      list.items.push(m[2]);
    } else if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) {
      flushPara();
      if (!list || list.type !== "ul") { flushList(); list = { type: "ul", items: [] }; }
      list.items.push(m[1]);
    } else if (list && /^\s{2,}/.test(raw)) {
      list.items[list.items.length - 1] += " " + line.trim();
    } else {
      flushList();
      para.push(line);
    }
  }
  flushPara();
  flushList();
  if (code) out.push(`<pre><code>${esc(code.join("\n"))}</code></pre>`);
  return out.join("");
}

/* ---------------- agent trace: what 寻寻 did, in its own words ---------------- */

/** One of several phrasings, fixed per answer (repaints must not reshuffle it). */
function pick(seed, options) {
  let h = 0;
  for (const ch of String(seed)) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return options[h % options.length];
}

/** A step as the puppy tells it, built from the event's real numbers; the technical name
    stays in the tooltip. `seed` picks the phrasing so it varies between answers. */
function describeStep(ev, seed = "") {
  const s = `${seed}:${ev.node}`;
  switch (ev.node) {
    case "analyze_query": {
      const what = {
        knowledge: "去资料里找答案", find_file: "帮你找文件", chitchat: "陪你聊两句", clarification: "先问清楚你的意思",
        summarize_file: "把整份文件读一遍", table_query: "数一数这张表",
      }[ev.intent] || "看看怎么帮你";
      const q = ev.rewritten_query ? `<em>要找</em> <code>${esc(ev.rewritten_query)}</code>` : "";
      return { label: `${pick(s, ["听懂了", "明白", "收到"])}：${what}`, detail: q, tip: `意图识别：${INTENT_LABEL[ev.intent] || ev.intent || ""}` };
    }
    case "retrieve": {
      const n = ev.retrieved ?? 0;
      const tip = "混合检索：向量 + 关键词 → 融合排序 → 精排";
      if (ev.retrieval_attempts > 1) return { label: n ? `换个方向又找了一遍，叼回 ${n} 段` : "换个方向又找了一遍，还是没找到", detail: "", tip };
      if (!n) return { label: "翻了一圈，没找到相关的内容", detail: "", tip, bad: true };
      return { label: pick(s, [`翻遍资料，叼回 ${n} 段相关内容`, `嗅了一圈，找到 ${n} 段相关内容`, `从资料里叼回 ${n} 段内容`]), detail: "", tip };
    }
    case "rewrite_search":
      return { label: pick(s, ["这些不太对路，换个说法再找", "不够相关，换个问法再找找"]), detail: "", bad: true, tip: "相关性不足，改写查询" };
    case "build_context":
      return { label: pick(s, ["把前后文一起叼过来读一读", "连上下文一起读一遍"]), detail: "", tip: "还原父块，构建上下文" };
    case "search_files":
      return { label: `翻了翻文件夹，找到 ${ev.found ?? 0} 个文件`, detail: "", tip: "元数据过滤 → 内容检索 → 按文件聚合" };
    case "read_whole_file":
      return { label: "从头到尾读完了", detail: "", tip: "逐段做笔记，再按笔记写总结" };
    case "query_table":
      return { label: "把表格数了一遍", detail: "", tip: "按固定操作在本机计算，不生成代码" };
    case "generate_answer":
      if (ev.model_error) return { label: "写到一半，模型服务断开了", detail: "", bad: true };
      return { label: pick(s, ["开始写回答", "动笔写回答"]), detail: "" };
    case "fact_check":
      if (ev.check_skipped && ev.check_mode === "rules") return { label: "逐句核对没能完成，只核对了数字和名称", detail: "<em>出处请自己点开看看</em>", bad: true, tip: "事实校验未完成，已按规则核对" };
      if (ev.check_skipped) return { label: "核对时网络开小差了", detail: "<em>回答没核对完，出处请自己点开看看</em>", bad: true, tip: "事实校验未完成" };
      if (ev.check_mode === "rules") return { label: "有数字或名称在原文里找不到", detail: "", bad: true, tip: "规则核对未通过" };
      if (ev.check_mode === "citations") {
        return ev.grounded
          ? { label: "看过了，每处出处都对得上", detail: "", tip: "快速模式：只核对引用出处" }
          : { label: "有几句没标出处", detail: "", bad: true, tip: "快速模式：只核对引用出处" };
      }
      return ev.grounded
        ? { label: pick(s, ["逐句对照原文，没问题", "每句都和原文对过了"]), detail: "", tip: "事实校验通过" }
        : { label: "有几句和原文对不上", detail: "", bad: true, tip: "事实校验未通过" };
    case "refuse":
      return { label: "资料里没找到可靠的依据，不乱说", detail: "", bad: true };
    case "clarification":
      return { label: "需要你再说清楚一点", detail: "" };
    case "chitchat":
      return { label: "陪你聊两句", detail: "" };
    case "finish":
      return null;
    default:
      return { label: esc(ev.node), detail: "" };
  }
}

function renderTrace(steps, { running, elapsed, open, label }) {
  const items = steps.map((s, i) => {
    const last = i === steps.length - 1;
    const cls = s.bad ? "bad" : running && last ? "running" : "done";
    return `<li class="${cls}"${s.tip ? ` title="${esc(s.tip)}"` : ""}><b>${s.label}</b>${s.detail ? " " + s.detail : ""}</li>`;
  });
  const head = running
    ? `${Icons.mascot("working")}<span class="shimmer">${label || `${PET}正在翻资料…`}</span>`
    : `${Icons.mascot()}${PET}跑了 ${steps.length} 步${elapsed ? ` · ${elapsed.toFixed(1)} 秒` : ""}${ICONS.chev}`;
  return `<div class="trace${open ? " open" : ""}">
    <button class="trace-head" type="button" aria-expanded="${!!open}">${head}</button>
    <ul class="trace-steps">${items.join("")}</ul>
  </div>`;
}

/* ---------------- message rendering ---------------- */

function userMessage(text) {
  const el = document.createElement("div");
  el.className = "msg msg-user";
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  return el;
}

function botShell(id) {
  const el = document.createElement("div");
  el.className = "msg msg-bot";
  el.dataset.bot = id;
  el.innerHTML = '<div class="bot-body"></div>';
  return el;
}

function sourceCard(s) {
  const lost = LOST.has(s.status) ? `<em class="lost">· ${esc(STATUS_LABEL[s.status])}</em>` : "";
  const ext = (s.source.match(/\.[^.]+$/) || [""])[0];
  return `<button type="button" class="source${s.cited ? "" : " uncited"}" data-n="${s.n}" data-source="${esc(s.source)}" data-doc="${esc(s.doc_id)}" data-locator="${esc(s.locator)}" data-quote="${esc(s.quote || "")}" title="${LOST.has(s.status) ? "原文件已丢失，仍可查看已解析的文本" : "查看原文"}">
      <span class="n">${s.n}</span>${FileRows.icon(ext)}
      <span class="src-text"><strong title="${esc(s.source)}">${esc(s.source)}</strong><span class="src-loc" title="${esc(s.locator)}">${esc(locatorLabel(s.locator) || "—")}${lost}</span></span>
    </button>`;
}

/* ---------------- asking a failed question again (retry.js) ---------------- */

// The one failed answer counting down to asking again by itself.
const autoRetry = { botId: "", until: 0, timer: 0, tick: 0 };

/** "重新提问" under the latest answer when it failed; "去设置" when the key or the balance is the cause. */
function retryRow(m) {
  if (!Retry.failed(m)) return "";
  const conv = currentConversation();
  if (!conv || conv.messages[conv.messages.length - 1] !== m) return ""; // only the latest answer
  const offer = Retry.plan(m.errorKind, { retryable: m.retryable ?? null, autoUsed: !!m.retried });
  if (offer.action === "settings") {
    return `<div class="retry-row"><button type="button" class="btn small" data-act="retry-settings">去设置</button></div>`;
  }
  if (offer.action !== "retry") return "";
  const counting = autoRetry.botId === m.id;
  return `<div class="retry-row">
      <button type="button" class="btn small primary" data-act="retry-ask" data-bot="${esc(m.id)}">${ICONS.rewrite}重新提问</button>
      ${counting ? `<span class="retry-auto"><span class="retry-auto-text">${esc(Retry.countdown((autoRetry.until - Date.now()) / 1000))}</span>
        <button type="button" class="retry-cancel" data-act="retry-cancel">取消</button></span>` : ""}
    </div>`;
}

function cancelAutoRetry() {
  clearTimeout(autoRetry.timer);
  clearInterval(autoRetry.tick);
  const id = autoRetry.botId;
  autoRetry.botId = "";
  const bot = id && currentConversation()?.messages.find((m) => m.id === id);
  if (bot) paint(bot);
}

/** A failed answer the service may give next time: count down, then ask again once. */
function scheduleAutoRetry(conv, bot) {
  const offer = Retry.plan(bot.errorKind, { retryable: bot.retryable ?? null, autoUsed: !!bot.retried });
  if (offer.action !== "retry" || !offer.auto || conv.id !== state.current || isRunning()) return;
  cancelAutoRetry();
  autoRetry.botId = bot.id;
  autoRetry.until = Date.now() + offer.auto * 1000;
  autoRetry.tick = setInterval(() => {
    const line = els.thread.querySelector(`.msg-bot[data-bot="${bot.id}"] .retry-auto-text`);
    if (line) line.textContent = Retry.countdown((autoRetry.until - Date.now()) / 1000);
  }, 500);
  autoRetry.timer = setTimeout(() => {
    if (autoRetry.botId !== bot.id) return;
    cancelAutoRetry();
    retryAsk(bot.id);
  }, offer.auto * 1000);
  paint(bot);
}

/** Ask the latest, failed question again: the failed answer is replaced, the question is not repeated. */
function retryAsk(botId) {
  const conv = currentConversation();
  if (!conv || isRunning()) return;
  const index = conv.messages.findIndex((m) => m.id === botId);
  const bot = conv.messages[index];
  if (!bot || index !== conv.messages.length - 1 || !Retry.failed(bot)) return;
  cancelAutoRetry();
  conv.messages.splice(index, 1);
  els.thread.querySelector(`.msg-bot[data-bot="${botId}"]`)?.remove();
  ask(bot.question, { retry: true });
}

function fillBot(el, m, { running = false, typing = false } = {}) {
  const body = el.querySelector(".bot-body");
  body.querySelector(".deck")?._deck?.destroy();
  let html = "";
  if (m.steps?.length || running) {
    // Once a draft is streaming the trace folds away and only names the phase.
    html += renderTrace(m.steps || [], {
      running,
      elapsed: m.elapsed,
      open: running && !m.answer,
      label: m.waiting || (m.phase === "checking" ? `${PET}正在对照原文核对…` : m.phase === "generating" ? `${PET}正在写回答…` : m.reading || ""),
    });
  }
  if (m.notice) html += `<div class="meta notice"><span class="chip warn">${ICONS.alert}${esc(m.notice)}</span></div>`;

  if (m.error) {
    html += `<div class="error-box">${esc(m.error)}</div>`;
  } else if (m.done && m.intent === "find_file") {
    html += renderFileResults(m);
  } else if (m.answer) {
    // Drafts hide citation markers (complete or still being typed) until `final` links them.
    const { html: answerHtml, sources } = typing
      ? { html: renderMarkdown(Citations.strip(m.answer)), sources: [] }
      : linkCitations(m.answer, m.citations);
    html += `<div class="answer${typing && m.phase !== "checking" ? " typing" : ""}">${answerHtml}</div>`;

    if (m.done) {
      const chips = [];
      if (m.refused && m.modelError) chips.push(`<span class="chip warn">${ICONS.alert}模型服务出错，没能写出回答</span>`);
      else if (m.refused) {
        // A whole-file or table question that could not run says why in its own words.
        if (m.intent !== "table_query" && m.intent !== "summarize_file") chips.push(`<span class="chip warn">${ICONS.alert}资料里证据不足，没有回答</span>`);
      }
      else if (m.checkSkipped && m.checkMode === "rules") chips.push(`<span class="chip warn" title="模型核对没能完成，只按规则核对了回答里的数字和名称都在原文里">${ICONS.alert}只核对了数字和名称，请点开出处看看</span>`);
      else if (m.checkSkipped) chips.push(`<span class="chip warn" title="事实校验服务限流或中断，回答未经核对">${ICONS.alert}没核对完，请点开出处看看</span>`);
      else if (m.checkMode === "citations") {
        chips.push(m.grounded
          ? `<span class="chip" title="快速模式：已确认每处引用都指向检索到的原文，未逐句校验内容">${ICONS.check}出处已核对</span>`
          : `<span class="chip warn" title="快速模式：有内容没有标注出处，或引用不在检索到的原文里">${ICONS.alert}部分内容没有出处</span>`);
      }
      else if (m.checkMode === "table") {
        if (m.grounded) chips.push(`<span class="chip ok" title="在本机按原表逐行计算，没有经过模型">${ICONS.shield}按原表算出</span>`);
      }
      else if (m.grounded && m.coverage?.checked_against === "notes") chips.push(`<span class="chip ok" title="这个文件很长，回答已逐句对照从原文整理的笔记">${ICONS.shield}已对照笔记</span>`);
      else if (m.grounded) chips.push(`<span class="chip ok" title="已逐句对照原文">${ICONS.shield}已对照原文</span>`);
      else if (m.intent === "knowledge" || (m.intent === "summarize_file" && m.coverage?.notes)) chips.push(`<span class="chip warn">${ICONS.alert}有内容和原文对不上</span>`);
      if (m.attempts > 1) chips.push(`<span class="chip">${ICONS.search}找了 ${m.attempts} 轮</span>`);
      if (m.rewritten && m.rewritten !== m.question) chips.push(`<span class="chip" title="改写后的检索词">${ICONS.rewrite}${esc(m.rewritten)}</span>`);
      if (chips.length) html += `<div class="meta">${chips.join("")}</div>`;

      const cited = sources.filter((s) => s.cited);
      const rest = sources.filter((s) => !s.cited);
      const shown = cited.length ? cited : rest;
      const hidden = cited.length ? rest : [];
      html += `<div class="msg-actions">
          <button type="button" class="btn plain" data-act="copy">${ICONS.copy}复制</button>
          ${shown.some((s) => s.doc_id) ? `<button type="button" class="btn plain" data-act="scope-sources">${ICONS.scope}只问这些来源</button>` : ""}
        </div>`;
      if (sources.length) {
        const deck = shown.length > 1;
        html += `<div class="sources">
          <div class="sources-head"><span class="sources-label">${cited.length ? "引用来源" : "检索到的片段"} · ${shown.length}</span>
            ${deck ? `<button type="button" class="deck-toggle">${m.deckOpen ? "收起" : "展开"}</button>` : ""}</div>
          <div class="deck">${shown.map(sourceCard).join("")}</div>
          ${hidden.length ? `<button class="more-sources" type="button">另有 ${hidden.length} 个检索片段没被引用 ›</button>
            <div class="source-grid" hidden>${hidden.map(sourceCard).join("")}</div>` : ""}
        </div>`;
      }
    }
  }
  if (!running) html += retryRow(m);
  body.innerHTML = html;
  const deckEl = body.querySelector(".deck");
  if (deckEl) {
    deckEl._deck = new UI.Deck(deckEl, {
      collapsed: !m.deckOpen,
      onToggle: (open) => {
        m.deckOpen = open;
        const t = body.querySelector(".deck-toggle");
        if (t) t.textContent = open ? "收起" : "展开";
      },
    });
  }
  const wheelEl = body.querySelector(".file-results [data-time-wheel]");
  if (wheelEl) mountTimeWheel(wheelEl, m);
  if (m.done) {
    prepareQueue.add([
      ...(m.citations || []).map((c) => ({ id: c.doc_id, name: c.source })),
      // Only the rows on screen; a long list stays folded.
      ...(m.files || []).slice(0, FileRows.COLLAPSED).map((f) => ({ id: f.file_id, name: f.name })),
    ]);
  }
}

/* ---------------- file results ---------------- */

/** Find-file results: compact rows (filerow.js), same-name files stacked, folded past five;
    a time wheel narrows them by when the files changed. */
const TIME_FILTERS = [
  { label: "全部", time: "" },
  { label: "今天", time: "今天" },
  { label: "7 天", time: "最近7天" },
  { label: "30 天", time: "最近30天" },
];

/** The wheel's choices; a time the question named itself ("上周的 PDF") comes first after 全部. */
function timeOptions(m) {
  if (m.ownTime === undefined) m.ownTime = m.fileSearch?.filters?.time_phrase || "";
  const own = m.ownTime && !TIME_FILTERS.some((o) => o.time === m.ownTime) ? [{ label: m.ownTime, time: m.ownTime }] : [];
  return [TIME_FILTERS[0], ...own, ...TIME_FILTERS.slice(1)];
}

function fileListHtml(m) {
  const files = m.files || [];
  if (!files.length) return `<div class="frows-empty">${Icons.mascot()}<span>${m.timeIndex ? "换个时间范围试试？" : `${PET}翻遍了也没找到。`}</span></div>`;
  return FileRows.list(files, { filters: m.fileSearch?.filters });
}

function renderFileResults(m) {
  const head = (m.answer || "").split("\n")[0] || "没有找到符合条件的文件。";
  return `<div class="file-results" data-bot="${esc(m.id)}">
      <div class="fr-head"><p class="fr-title">${esc(head)}</p>
        <div class="fr-time"><span id="frt-${esc(m.id)}">时间</span><div class="wheel compact" data-time-wheel aria-labelledby="frt-${esc(m.id)}"></div></div>
      </div>
      <div class="fr-list">${fileListHtml(m)}</div>
    </div>`;
}

function mountTimeWheel(el, m) {
  const options = timeOptions(m);
  const index = m.timeIndex ?? Math.max(0, options.findIndex((o) => o.time === m.ownTime));
  new UI.Wheel(el, {
    items: options.map((o) => ({ label: o.label })),
    index,
    label: "按修改时间筛选",
    onChange: (_, i) => refilterFiles(el.closest(".file-results"), m, i),
  });
}

/** Search again with another time range (rules only, no model call), keeping the question. */
async function refilterFiles(box, m, i) {
  const option = timeOptions(m)[i];
  m.timeIndex = i;
  let q = m.question || "";
  if (m.ownTime && option.time !== m.ownTime) q = q.replace(m.ownTime, " ");
  const params = new URLSearchParams({ q, limit: "100" });
  if (option.time) params.set("time", option.time);
  const list = box.querySelector(".fr-list");
  list.style.opacity = "0.5";
  try {
    const data = await api(`/files/search?${params}`);
    if (m.timeIndex !== i) return;
    m.files = data.files || [];
    m.fileSearch = { ...(m.fileSearch || {}), elapsed_ms: data.elapsed_ms, filters: data.filters, total: data.total };
    prepareQueue.add(m.files.slice(0, FileRows.COLLAPSED).map((f) => ({ id: f.file_id, name: f.name })));
    const total = data.total ?? m.files.length;
    m.answer = m.files.length
      ? `共 ${total} 个文件${total > m.files.length ? `，列出 ${m.files.length} 个` : ""}：`
      : `${option.label === "全部" ? "" : `${option.label}内`}没有符合条件的文件。`;
    box.querySelector(".fr-title").textContent = m.answer;
    list.innerHTML = fileListHtml(m);
    list.animate?.([{ opacity: 0, transform: "translateY(4px)" }, { opacity: 1, transform: "none" }], { duration: 260, easing: "ease-out" });
    saveStore();
  } catch (err) {
    toast(`筛选失败：${err.message || err}`);
  } finally {
    list.style.opacity = "";
  }
}

/** Ask the next questions about one file only. */
function askAboutFile(id, name) {
  setScope({ doc_ids: [id], label: name });
  els.input.focus();
  toast(`之后的问题只在「${name}」中检索`);
}

/* ---------------- question scope ---------------- */

/** Where the next question searches: {doc_ids, folder, label}; null = the whole library. */
function currentScope() {
  const conv = currentConversation();
  return conv ? conv.scope || null : state.pendingScope;
}

function setScope(scope) {
  const conv = currentConversation();
  if (conv) { conv.scope = scope; saveStore(); } else state.pendingScope = scope;
  renderScope();
}

function renderScope() {
  const scope = currentScope();
  const bar = $("#scopeBar");
  bar.hidden = !scope;
  els.composer.classList.toggle("scoped", !!scope);
  if (!scope) return;
  const what = scope.folder ? "文件夹" : "文件";
  bar.innerHTML = `<span class="scope-chip" title="${esc(scope.folder || scope.label)}">
      ${scope.folder ? ICONS.folder : ICONS.search}仅在${what}内提问：<b>${esc(scope.label)}</b>
      <button type="button" class="scope-clear" aria-label="取消范围限定" title="取消范围限定，在整个知识库中提问">${ICONS.x}</button>
    </span>`;
}

const scopePicker = { roots: [], folders: [] };

async function openScopePicker() {
  const pop = $("#scopePop");
  if (!pop.hidden) { pop.hidden = true; return; }
  pop.hidden = false;
  $("#scopeFilter").value = "";
  $("#scopeFilter").focus();
  renderScopeList();
  try {
    const [files, roots] = await Promise.all([api("/files?status=indexed,missing,offline&limit=5000"), api("/watch-roots")]);
    const folders = new Set();
    for (const f of files.files || []) {
      const dir = (f.path || "").replace(/[\\/][^\\/]*$/, "");
      if (dir) folders.add(dir);
    }
    for (const r of roots.roots || roots || []) if (r.path) folders.add(r.path);
    scopePicker.files = (files.files || []).map((f) => ({ id: f.file_id, name: f.name, path: f.path }));
    scopePicker.folders = [...folders].sort();
  } catch (_) {
    if (!state.docs.length) await loadDocs();
    scopePicker.files = state.docs.map((d) => ({ id: d.doc_id, name: d.source, path: d.path || "" }));
    scopePicker.folders = [];
  }
  renderScopeList();
}

function renderScopeList() {
  const q = $("#scopeFilter").value.trim().toLowerCase();
  const match = (text) => !q || (text || "").toLowerCase().includes(q);
  const folders = (scopePicker.folders || []).filter(match).slice(0, 30);
  const files = (scopePicker.files || []).filter((f) => match(f.name) || match(f.path)).slice(0, 60);
  const rows = [
    `<li data-scope-all><b>整个知识库</b></li>`,
    ...folders.map((f) => `<li data-scope-folder="${esc(f)}">${ICONS.folder}<span><b>${esc(f.split(/[\\/]/).pop() || f)}</b><small>${esc(f)}</small></span></li>`),
    ...files.map((f) => `<li data-scope-file="${esc(f.id)}" data-name="${esc(f.name)}">${ICONS.search}<span><b>${esc(f.name)}</b><small>${esc(f.path)}</small></span></li>`),
  ];
  if (!scopePicker.files) rows.push('<li class="muted"><span class="spinner"></span>正在读取文件列表…</li>');
  $("#scopeList").innerHTML = rows.join("");
}

/* ---------------- state ---------------- */

const state = {
  conversations: [],
  // "loading" | "ready" | "failed": the sidebar must not read as empty while it doesn't know.
  historyState: "loading",
  historyError: "",
  // Shown in the history while a conversation can't be saved.
  saveIssue: "",
  // The conversation whose messages are being read to open it.
  opening: null,
  current: null,
  // conversation id -> { controller, bot } for answers still streaming
  running: new Map(),
  docs: [],
  // Scope chosen before the first question of a new conversation.
  pendingScope: null,
  // "回答后推荐追问" (Settings); read from the backend at start.
  followupsEnabled: true,
  // The conversation on screen when it was last drawn: leaving it cancels its follow-up request.
  shown: null,
  // A suggestion put in the input with Tab: sending it (edited or not) applies its scope.
  filled: null,
  // Shown in the status row after an answer failed at the model service.
  modelIssue: "",
};

function currentConversation() {
  return state.conversations.find((c) => c.id === state.current) || null;
}

function isRunning(convId = state.current) {
  return state.running.has(convId);
}

/** Streams keep updating their message object; it is only drawn while its conversation is on screen. */
function paint(bot, opts) {
  const el = els.thread.querySelector(`.msg-bot[data-bot="${bot.id}"]`);
  if (el) fillBot(el, bot, opts);
  return !!el;
}

function renderHistory() {
  const notes = [];
  if (state.saveIssue) notes.push(`<li class="history-note warn" title="${esc(state.saveIssue)}">${ICONS.alert}<span>对话没能保存，正在重试</span></li>`);
  if (state.historyState === "failed") {
    notes.push(`<li class="history-note warn" title="${esc(state.historyError)}">${ICONS.alert}<span>对话记录没读出来</span>
      <button type="button" class="history-retry" data-act="reload-history">重试</button></li>`);
  } else if (state.historyState === "loading" && !state.conversations.length) {
    notes.push('<li class="history-empty">正在读取…</li>');
  } else if (!state.conversations.length) {
    notes.push('<li class="history-empty">还没有对话</li>');
  }
  els.history.innerHTML = notes.join("") + state.conversations
    .map((c) => `<li class="${c.id === state.current ? "active" : ""}${isRunning(c.id) ? " live" : ""}" data-id="${esc(c.id)}">
        <button type="button" title="${esc(c.title)}">${esc(c.title)}</button>
        ${isRunning(c.id) ? `<span class="live-dots" title="${PET}正在回答"><i></i><i></i><i></i></span>` : ""}
        <button type="button" class="del" aria-label="删除对话">${ICONS.x}</button>
      </li>`)
    .join("");
}

function renderConversation() {
  noteShown();
  // Another conversation, or this one again: a countdown to asking again does not follow.
  if (autoRetry.botId) {
    clearTimeout(autoRetry.timer);
    clearInterval(autoRetry.tick);
    autoRetry.botId = "";
  }
  els.thread.querySelectorAll(".msg").forEach((n) => n.remove());
  const conv = currentConversation();
  setHome(!(conv && conv.messages.length));
  if (conv) {
    for (const m of conv.messages) {
      if (m.role === "user") els.thread.appendChild(userMessage(m.text));
      else {
        const el = botShell(m.id);
        const live = state.running.get(conv.id)?.bot === m;
        fillBot(el, m, live ? { running: true, typing: !!m.answer } : {});
        els.thread.appendChild(el);
      }
    }
  }
  renderHistory();
  renderScope();
  updateComposer();
  renderFollowups();
  requestAnimationFrame(() => scrollToBottom(true));
}

function newConversation(source) {
  return UI.circleTransition(source, () => {
    state.opening = null;
    state.current = null;
    state.pendingScope = null;
    renderConversation();
    showNextSuggestions();
    showView("chat");
  }).then(() => els.input.focus());
}

/* ---------------- chat ---------------- */

async function ask(question, { retry = false } = {}) {
  question = question.trim();
  if (!question || isRunning()) return;
  cancelAutoRetry();

  let conv = currentConversation();
  if (!conv) {
    conv = { id: uid(), title: question.slice(0, 40), messages: [], updated: Date.now(), scope: state.pendingScope };
    state.pendingScope = null;
    state.conversations.unshift(conv);
    state.current = conv.id;
    noteShown();
  } else {
    state.conversations = [conv, ...state.conversations.filter((c) => c !== conv)];
  }
  conv.updated = Date.now();

  // "重新提问" replaces the failed answer under the same question; the question is not repeated.
  if (!retry) conv.messages.push({ role: "user", text: question });
  const bot = { role: "bot", id: uid(), question, steps: [], answer: "", done: false, ...(retry ? { retried: true } : {}) };
  conv.messages.push(bot);
  // The question is on disk from the start: closing the app mid-answer doesn't lose it.
  saveStore(conv);
  followupRequests.cancel(conv.id);
  state.filled = null;

  setHome(false);
  if (!retry) els.thread.appendChild(userMessage(question));
  const el = botShell(bot.id);
  els.thread.appendChild(el);
  fillBot(el, bot, { running: true });
  scrollToBottom(true);

  if (!retry) {
    els.input.value = "";
    updatePlaceholder();
  }
  const started = performance.now();
  const controller = new AbortController();
  state.running.set(conv.id, { controller, bot });
  renderHistory();
  autosize();
  updateComposer();
  renderFollowups();

  try {
    const res = await apiFetch(`${API}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        thread_id: conv.id,
        ...(conv.scope ? { scope: { doc_ids: conv.scope.doc_ids || [], folder: conv.scope.folder || "" } } : {}),
        ...(retry ? { retry: true } : {}),
      }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error(await errorDetail(res));

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, cut);
        buffer = buffer.slice(cut + 2);
        const data = chunk.split("\n").filter((l) => l.startsWith("data:")).map((l) => l.slice(5).trimStart()).join("\n");
        if (data) handleEvent(JSON.parse(data), bot, started);
      }
    }
    if (!bot.done && !bot.error) throw new Error("连接意外中断");
  } catch (err) {
    if (bot.paintFrame) cancelAnimationFrame(bot.paintFrame);
    bot.paintFrame = 0;
    if (bot.drafting) bot.notice = "已停止，这段回答未经过事实校验";
    bot.drafting = false;
    bot.phase = "";
    if (err.name === "AbortError") {
      bot.error = bot.answer ? null : "已停止生成。";
      bot.done = true;
    } else {
      // A stream cut off mid-read rejects with a bare TypeError ("network error").
      const reason = err instanceof TypeError ? "与后台服务的连接中断了。" + BACKEND_UNREACHABLE.split("：")[1] : err.message || err;
      bot.error = `请求失败：${reason}`;
      // The backend could not be reached or answered with an error page: asking again may help.
      bot.errorKind = bot.errorKind || (err instanceof TypeError ? "network" : "unknown");
      bot.retryable = true;
    }
    bot.elapsed = (performance.now() - started) / 1000;
    paint(bot);
  } finally {
    state.running.delete(conv.id);
    // The status row says so while the model service keeps failing; an answer clears it.
    const issue = bot.modelError ? "模型连接失败" : bot.done && !bot.error ? "" : state.modelIssue;
    if (issue !== state.modelIssue) { state.modelIssue = issue; renderStatus(); }
    renderHistory();
    updateComposer();
    saveStore(conv);
    if (conv.id === state.current) scrollToBottom();
    else if (state.conversations.includes(conv)) toast(`「${conv.title}」的回答已完成`);
  }
  if (!state.conversations.includes(conv)) return;
  // Overloaded, rate limited, unreachable: count down and ask again once by itself. First, since
  // a refusal for a failed model call is also `done` with no `error`.
  if (Retry.failed(bot)) scheduleAutoRetry(conv, bot);
  // After `final`, not before: the answer is on screen while the follow-ups are written.
  else if (bot.done && !bot.error) requestFollowups(conv, bot);
}

/** Draft tokens arrive hundreds of times a second; repaint at most once per frame. */
function scheduleDraftPaint(bot) {
  if (bot.paintFrame) return;
  bot.paintFrame = requestAnimationFrame(() => {
    bot.paintFrame = 0;
    if (paint(bot, { running: true, typing: true })) scrollToBottom();
  });
}

function handleEvent(ev, bot, started) {
  if (bot.paintFrame && ev.event !== "draft") {
    cancelAnimationFrame(bot.paintFrame);
    bot.paintFrame = 0;
  }
  // A wait for the model service ends with whatever comes next.
  if (bot.waiting && ev.event !== "progress") bot.waiting = "";
  switch (ev.event) {
    case "node": {
      const step = describeStep(ev, bot.id);
      if (step) bot.steps.push(step);
      if (ev.node === "generate_answer" && ev.model_error) {
        // Generation broke off midway: the partial draft is withdrawn, the reason shown.
        bot.modelError = ev.model_error;
        if (bot.drafting) { bot.answer = ""; bot.drafting = false; bot.phase = ""; }
        bot.notice = bot.drafted ? "生成中断，初稿已撤回" : null;
      } else if (bot.drafting && ev.node === "generate_answer") bot.phase = "checking";
      if (bot.drafting && (ev.node === "rewrite_search" || ev.node === "refuse")) {
        // The fact check rejected the draft: withdraw it before anything else is shown.
        bot.answer = "";
        bot.drafting = false;
        bot.phase = "";
        bot.notice = ev.node === "rewrite_search"
          ? "初稿未通过事实校验，已撤回，正在重新检索"
          : "初稿未通过事实校验，已撤回";
      }
      paint(bot, { running: true, typing: !!bot.answer });
      break;
    }
    case "progress":
      if (ev.kind === "reading") {
        bot.reading = `${PET}正在从头读这份文件：${ev.done}/${ev.total} 段…`;
        paint(bot, { running: true, typing: !!bot.answer });
      } else if (ev.kind === "waiting") {
        // The model service is rate limited: the answer or its check waits instead of being skipped.
        bot.waiting = `${PET}模型这会儿有点忙，排队等 ${ev.seconds || ""} 秒再接着${bot.phase === "checking" ? "核对" : "写"}…`;
        // A draft cut off by the failure is written again from the start, not appended to.
        bot.drafting = false;
        paint(bot, { running: true, typing: !!bot.answer });
      }
      return;
    case "draft":
      if (!bot.drafting) {
        bot.drafting = bot.drafted = true;
        bot.answer = "";
        bot.phase = "generating";
        if (bot.notice?.includes("重新检索")) bot.notice = "初稿未通过事实校验，已按新的检索结果重写";
      }
      bot.answer += ev.content || "";
      scheduleDraftPaint(bot);
      return;
    case "token":
      if (bot.drafted) break; // The draft already showed this text; `final` replaces it.
      bot.answer += ev.content || "";
      paint(bot, { running: true, typing: true });
      break;
    case "final":
      bot.drafting = false;
      bot.phase = "";
      if (bot.notice && !ev.refused) bot.notice = null;
      Object.assign(bot, {
        answer: ev.answer || bot.answer || "",
        citations: ev.citations || [],
        grounded: !!ev.grounded,
        refused: !!ev.refused,
        checkSkipped: !!ev.fact_check_skipped,
        checkMode: ev.check_mode || "",
        intent: ev.intent,
        rewritten: ev.rewritten_query,
        attempts: ev.retrieval_attempts,
        files: ev.files || [],
        fileSearch: ev.file_search || null,
        coverage: ev.coverage || null,
        // The model or the search failed (not "no evidence"): the answer offers asking again.
        errorKind: ev.error_kind || "",
        retryable: !!ev.retryable,
        reading: "",
        elapsed: (performance.now() - started) / 1000,
        done: true,
      });
      paint(bot);
      break;
    case "error":
      bot.error = ev.message || "问答服务暂时不可用。";
      bot.errorKind = ev.kind || "unknown";
      bot.retryable = ev.retryable !== false;
      bot.elapsed = (performance.now() - started) / 1000;
      paint(bot);
      break;
  }
  if (els.thread.querySelector(`.msg-bot[data-bot="${bot.id}"]`)) scrollToBottom();
}

function updateComposer() {
  const busy = isRunning();
  els.composer.classList.toggle("busy", busy);
  els.send.setAttribute("aria-label", busy ? "停止" : "发送");
  updateSend();
}

function updateSend() {
  els.send.disabled = !isRunning() && !els.input.value.trim();
}

function autosize() {
  els.input.style.height = "auto";
  // An empty box keeps one line even when the grey suggestion is longer than it.
  const height = els.input.value ? els.input.scrollHeight : 0;
  els.input.style.height = height ? Math.min(height, 200) + "px" : "";
  els.input.style.overflowY = height > 200 ? "auto" : "hidden";
  updateSend();
}

/* ---------------- follow-ups (回答后推荐追问) ---------------- */

const followupRequests = new Followups.Requests(async (body, signal) => {
  const res = await apiFetch(`${API}/followups`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json();
});

function followupOpts(conv = currentConversation()) {
  return { running: !!conv && isRunning(conv.id), enabled: state.followupsEnabled };
}

function ghostFollowup() {
  return Followups.ghost(currentConversation(), followupOpts(), els.input.value);
}

/** Leaving a conversation drops its pending follow-up request. */
function noteShown() {
  if (state.shown === state.current) return;
  if (state.shown) followupRequests.cancel(state.shown);
  state.filled = null;
  state.shown = state.current;
}

/** Ask for the next questions once an answer is final; typing, a new question or leaving cancels it. */
async function requestFollowups(conv, bot) {
  if (!state.followupsEnabled) return;
  const data = await followupRequests.start(conv.id, { thread_id: conv.id, question: bot.question, n: 4 });
  if (!data) return;
  if (data.enabled === false) { state.followupsEnabled = false; return; }
  bot.followupsAsked = true;
  const last = conv.messages[conv.messages.length - 1];
  if (last !== bot || !(data.items || []).length) { saveStore(conv); return; }
  bot.followups = { items: data.items, scenario: data.scenario || "", dismissed: false };
  saveStore(conv);
  if (conv.id !== state.current) return;
  // Read before the chips go in: a reader at the end of the answer keeps seeing the end.
  const s = els.scroller;
  const atEnd = s.scrollHeight - s.scrollTop - s.clientHeight < 160;
  renderFollowups();
  if (atEnd) scrollToBottom(true);
}

/** An older answer without follow-ups gets them once, when its conversation is opened. */
function followupsForOpened(conv) {
  const last = conv.messages[conv.messages.length - 1];
  if (!state.followupsEnabled || isRunning(conv.id) || !last || last.role !== "bot") return;
  if (!last.done || last.error || last.followups || last.followupsAsked) return;
  requestFollowups(conv, last);
}

function followupChip(item, i) {
  const tag = item.label ? `<span class="fu-tag">${esc(item.label)}</span>` : "";
  const title = item.scope_action === "set" ? `只在「${item.scope?.label || ""}」中提问`
    : item.scope_action === "clear" ? "在整个知识库中提问" : "";
  return `<button type="button" class="fu-chip" data-i="${i}"${title ? ` title="${esc(title)}"` : ""}>${ICONS.followup}${tag}<span>${esc(item.text)}</span></button>`;
}

/** Alternatives below the last answer only, and the grey suggestion in the input. */
function renderFollowups() {
  els.thread.querySelectorAll(".followups").forEach((n) => n.remove());
  const conv = currentConversation();
  const bot = Followups.host(conv, followupOpts(conv));
  const rest = Followups.chips(conv, followupOpts(conv));
  if (bot && rest.length) {
    const body = els.thread.querySelector(`.msg-bot[data-bot="${bot.id}"] .bot-body`);
    body?.insertAdjacentHTML("beforeend", `<div class="followups" data-bot="${esc(bot.id)}">${rest.map((item, n) => followupChip(item, n + 1)).join("")}</div>`);
  }
  updatePlaceholder();
}

/** Esc: the grey suggestion goes for this answer (the alternatives stay). */
function dismissGhost() {
  followupRequests.cancel(state.current);
  const bot = Followups.host(currentConversation(), followupOpts());
  if (bot && !bot.followups.dismissed) {
    bot.followups.dismissed = true;
    saveStore();
  }
  updatePlaceholder();
}

function applyFollowupScope(item) {
  const scope = Followups.scopeChange(item);
  if (scope === undefined) return;
  const current = currentScope();
  if (scope === null) {
    if (current) { setScope(null); toast("这次在整个知识库中提问"); }
  } else if (JSON.stringify(current?.doc_ids || []) !== JSON.stringify(scope.doc_ids)) {
    setScope(scope);
    toast(`之后的问题只在「${scope.label}」中检索`);
  }
}

function sendFollowup(item) {
  if (!item || isRunning()) return;
  applyFollowupScope(item);
  els.input.value = "";
  ask(item.text);
}

/** Tab / →: the suggestion goes into the input to be edited; sending it keeps its scope. */
function fillFollowup(item) {
  els.input.value = item.text;
  state.filled = { conv: state.current, item };
  autosize();
  els.input.setSelectionRange(item.text.length, item.text.length);
  updatePlaceholder();
}

/** Send what is in the input; a suggestion filled in with Tab brings its scope along. */
function submitInput() {
  const filled = state.filled?.conv === state.current ? state.filled.item : null;
  state.filled = null;
  if (filled && els.input.value.trim()) applyFollowupScope(filled);
  ask(els.input.value);
}

/* ---------------- opening files: 寻寻 fetches, prepared ahead of time ---------------- */

/** Word files about to be shown get their preview prepared in the background (one at a
    time on the backend), so opening them rarely waits. */
const prepareQueue = new AppStatus.PrepareQueue((ids) => postJson("/documents/prepare", { doc_ids: ids }));

/** 寻寻 fetching a file: paw prints, a line that changes the longer it takes, a blank page. */
function fetchingHtml() {
  return `<div class="fetching" data-since="${Date.now()}" role="status">
      <div class="fetch-head">${Icons.mascot("working")}<span class="paws" aria-hidden="true">${Icons.svg("paw").repeat(4)}</span><span class="fetch-text">${AppStatus.fetchLine(0)}</span></div>
      <div class="skeleton-page" aria-hidden="true"><i class="h"></i><i></i><i></i><i class="s"></i><i class="gap"></i><i></i><i></i><i></i><i class="s"></i><i class="gap"></i><i></i><i class="s"></i></div>
    </div>`;
}

let fetchTicker = 0;
function showFetching(el) {
  el.innerHTML = fetchingHtml();
  if (fetchTicker) return;
  fetchTicker = setInterval(() => {
    const nodes = document.querySelectorAll(".fetching");
    if (!nodes.length) { clearInterval(fetchTicker); fetchTicker = 0; return; }
    nodes.forEach((n) => {
      const line = AppStatus.fetchLine(Date.now() - Number(n.dataset.since));
      const text = n.querySelector(".fetch-text");
      if (text.textContent !== line) text.replaceWith(Object.assign(document.createElement("span"), { className: "fetch-text", textContent: line }));
    });
  }, 300);
}

/* ---------------- documents ---------------- */

const lib = {
  filter: "",
  // Document type filter ("简历", "申报书"…; the file cards, see /library/types).
  docType: "",
  // How much of the list is shown ("显示更多") and the search box; see LibraryUI.
  fileLimit: LibraryUI.FILE_PAGE,
  search: "",
  filesStale: false,
  filesLoadedAt: 0,
  files: [],
  groups: [],
  roots: [],
  status: null,
  pollTimer: 0,
  preview: null,
  // Files ticked in the list (tri-state per folder) and folders folded shut.
  selection: new UI.Selection(),
  closed: new Set(),
  // The last change the status row was told about (/library/status?since=), how many lost
  // files the "找不到的文件" list shows (-1: reload it), and whether a batch is being added.
  changeSeq: 0,
  lostShown: -1,
  importing: false,
};

async function api(path, options = {}) {
  const res = await apiFetch(`${API}${path}`, options);
  if (!res.ok) throw new Error(await errorDetail(res));
  return res.json().catch(() => ({}));
}

function postJson(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

function fmtSize(bytes) {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function fmtNum(n) {
  return Number(n || 0).toLocaleString();
}

function badge(status) {
  return `<span class="state ${esc(status)}">${esc(STATUS_LABEL[status] || status)}</span>`;
}

/** "10月23日" / "今天 09:12" for a file's modified time (seconds since the epoch or ISO). */
function fmtTime(value) {
  if (value == null || value === "") return "";
  const date = new Date(typeof value === "number" ? value * 1000 : value);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  if (date.toDateString() === now.toDateString()) return `今天 ${pad(date.getHours())}:${pad(date.getMinutes())}`;
  if (date.getFullYear() === now.getFullYear()) return `${date.getMonth() + 1}月${date.getDate()}日`;
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

/** Searchable documents (citations look-up and the sidebar count), plus the library page. */
async function loadDocs() {
  try {
    state.docs = (await api("/documents")).documents || [];
    els.docCount.textContent = state.docs.length;
    renderHomeHead();
    if (document.querySelector("#view-library.active")) await Promise.all([loadFiles(), pollStatus()]);
    return true;
  } catch (err) {
    els.docList.innerHTML = `<p class="doc-empty">知识库暂时打不开：${esc(err.message || err)}</p>`;
    els.docCount.textContent = "–";
    return false;
  }
}

async function loadFiles() {
  try {
    const params = new URLSearchParams({ limit: String(lib.fileLimit) });
    if (lib.filter) params.set("status", lib.filter);
    if (lib.search) params.set("q", lib.search);
    if (lib.docType) params.set("doc_type", lib.docType);
    lib.filesLoadedAt = Date.now();
    lib.filesStale = false;
    const [data] = await Promise.all([api(`/files?${params}`), loadTypes()]);
    lib.files = data.files || [];
    lib.filesTotal = data.total;
    renderFiles(data.total);
  } catch (err) {
    els.docList.innerHTML = `<p class="doc-empty">文件列表暂时打不开：${esc(err.message || err)}</p>`;
  }
}

/** The type filter: the library's document types with their counts. Hidden while no file
    has a type; while cards are still being written it says how many files are left. */
async function loadTypes() {
  const wrap = $("#typeFilterWrap");
  const select = $("#typeFilter");
  let data;
  try {
    data = await api("/library/types");
  } catch {
    return;
  }
  const types = data.types || [];
  const named = types.filter((t) => t.type !== "其他");
  // Right after a big import no file has a type yet: the filter still shows how many wait.
  wrap.hidden = !named.length && !lib.docType && !data.pending;
  if (lib.docType && !types.some((t) => t.type === lib.docType)) types.unshift({ type: lib.docType, count: 0 });
  const all = data.pending ? `全部类型（${fmtNum(data.pending)} 个待判断）` : "全部类型";
  select.innerHTML = `<option value="">${all}</option>${types.map((t) =>
    `<option value="${esc(t.type)}">${esc(t.type)}（${fmtNum(t.count)}）</option>`).join("")}`;
  select.value = lib.docType;
  wrap.title = data.pending ? `还有 ${fmtNum(data.pending)} 个文件在判断类型${cardsLeft(data.seconds_left)}` : "";
}

/** "，约 3 小时" for the time the remaining document cards take (none when unknown). */
function cardsLeft(seconds) {
  if (!seconds) return "";
  const minutes = Math.max(1, Math.round(seconds / 60));
  return minutes < 90 ? `，约 ${minutes} 分钟` : `，约 ${Math.round(minutes / 60)} 小时`;
}

/** A file's type as a small button that opens its card. */
function typeChip(f) {
  const card = f.card;
  if (!card || !card.type) return "";
  return `<button type="button" class="type-chip" data-card="${esc(f.file_id)}" title="查看文件卡片">${esc(card.type)}</button>`;
}

/** The file card: what kind of document it is, who or what it is about, its time, one line. */
function showCard(fileId) {
  const f = lib.files.find((item) => item.file_id === fileId);
  const card = f?.card;
  if (!card) return;
  const type = card.label ? `${card.type}（${card.label}）` : card.type;
  const note = card.by_name && card.has_card
    ? `类型按文件名判断。卡片里写的是“${card.card_type || "其他"}”，以文件名为准。`
    : card.by_name ? "类型按文件名判断，文件卡片还没写好。"
    : "由问答模型读文件名和开头约 900 字写成，可能有误。";
  UI.infoSheet({
    title: f.name,
    rows: [["类型", type], ["关于", card.subject], ["时间", card.period], ["摘要", card.summary]],
    note,
  });
}

/** Files grouped by the folder they sit in; imported copies form one group. */
function fileGroups(files) {
  const roots = new Map(lib.roots.map((r) => [r.root_id, r]));
  const groups = new Map();
  for (const f of files) {
    const upload = f.origin === "upload";
    const dir = upload ? "" : (f.path || "").replace(/[\\/][^\\/]*$/, "");
    const key = upload ? "::uploads" : dir;
    if (!groups.has(key)) {
      const root = roots.get(f.root_id);
      let name = upload ? "导入的文件" : dir.split(/[\\/]/).pop() || dir;
      let where = upload ? "复制保存在数据目录" : dir;
      if (root && dir.toLowerCase().startsWith(root.path.toLowerCase())) {
        const rootName = root.path.split(/[\\/]/).filter(Boolean).pop() || root.path;
        const rest = dir.slice(root.path.length).split(/[\\/]/).filter(Boolean);
        name = rest.length ? rest[rest.length - 1] : rootName;
        where = [rootName, ...rest.slice(0, -1)].join(" › ") || root.path;
        if (!rest.length) where = root.path;
      }
      // Folder-level choices (只按文件名 / 不收录) need a watched folder around it.
      const inRoot = !upload && root && dir.toLowerCase().startsWith(root.path.toLowerCase());
      groups.set(key, { key, name, where, files: [], path: upload ? "" : dir, root: inRoot ? root : null });
    }
    groups.get(key).files.push(f);
  }
  return [...groups.values()];
}

function renderFiles(total) {
  lib.groups = fileGroups(lib.files);
  lib.selection.setGroups(lib.groups.map((g, i) => ({ id: `g${i}`, items: g.files.map((f) => f.file_id) })));
  if (!lib.files.length) {
    els.docList.innerHTML = lib.search
      ? '<p class="doc-empty">没有找到匹配的文件。</p>'
      : lib.docType
      ? '<p class="doc-empty">没有这个类型的文件。</p>'
      : lib.filter
      ? '<p class="doc-empty">没有这个状态的文件。</p>'
      : `<div class="lib-empty">${Icons.mascot()}<span>${PET}还没有资料可以翻。添加一个文件夹，它会自己读完。</span></div>`;
    syncSelection();
    return;
  }
  const html = lib.groups.map((g, i) => {
    const rows = g.files.map((f) => {
      const ext = f.name.includes(".") ? `.${f.name.split(".").pop()}` : "";
      const lost = LOST.has(f.status);
      const note = (f.status === "failed" || f.status === "unsupported") && f.error
        ? `<span class="doc-error" title="${esc(f.error)}">${esc(f.error)}</span>`
        : f.notice === "scanned_pdf" ? '<span class="doc-note">扫描件，还没识别文字</span>'
        : f.notice === "ocr_unavailable" ? '<span class="doc-note">等本机识别组件下载后再读图中文字</span>'
        : `<span class="doc-path">${typeChip(f)}${esc(fmtTime(f.mtime) ? `修改于 ${fmtTime(f.mtime)}` : "")}</span>`;
      const actions = [];
      if (f.notice === "scanned_pdf") actions.push(`<button class="btn small" type="button" data-scan="${esc(f.file_id)}">识别文字</button>`);
      if (f.status === "missing") actions.push(`<button class="btn small" type="button" data-locate="${esc(f.file_id)}">重新定位</button>`);
      else if (!lost) actions.push(`<button class="icon-btn sm" type="button" data-reveal="${esc(f.file_id)}" title="在资源管理器中显示" aria-label="在资源管理器中显示">${ICONS.reveal}</button>`);
      if (!lost && f.status !== "unsupported") actions.push(keepButton(`data-keep="${esc(f.file_id)}"`, f.content_mode === "name", "sm"));
      if (f.status !== "unsupported") actions.push(`<button class="icon-btn sm danger" type="button" data-del="${esc(f.file_id)}" data-name="${esc(f.name)}" title="从知识库移除" aria-label="从知识库移除">${ICONS.trash}</button>`);
      return `<li class="tree-file${lost ? " lost" : ""}" data-file="${esc(f.file_id)}"${f.searchable ? ` data-id="${esc(f.file_id)}" title="${esc(f.path)}"` : ` title="${esc(f.path)}"`}>
        <input type="checkbox" class="check" data-file-check="${esc(f.file_id)}" aria-label="选择 ${esc(f.name)}">
        ${FileRows.icon(ext)}
        <div class="tf-main"><strong>${esc(f.name)}</strong>${note}</div>
        <span class="doc-size">${fmtSize(f.size)}</span>
        <span class="state-cell">${badge(f.status)}</span>
        <div class="doc-actions">${actions.join("")}</div>
      </li>`;
    }).join("");
    const closed = lib.closed.has(g.key);
    const mode = PrivacyUI.groupMode(g.files);
    return `<div class="tree-group${closed ? " closed" : ""}" data-group="${i}">
      <div class="tree-folder" data-toggle="${i}">
        <input type="checkbox" class="check" data-group-check="g${i}" aria-label="选择文件夹 ${esc(g.name)} 里的全部文件">
        ${ICONS.chev}${Icons.svg("folder", "ic-folder")}
        <span class="tf-name"><b>${esc(g.name)}</b><span title="${esc(g.where)}">${esc(g.where)}</span></span>
        <span class="tf-count">${mode === "name" ? "只按文件名 · " : ""}${g.files.length} 个</span>
        ${g.root ? keepButton(`data-folder-keep="${i}"`, mode === "name", "sm") : ""}
      </div>
      <ul class="tree-files">${rows}</ul>
    </div>`;
  });
  const more = LibraryUI.moreFiles(total, lib.files.length);
  if (more) {
    html.push(`<p class="doc-more">${esc(more.text)}${more.step ? ` <button type="button" class="btn small" id="filesMore">显示更多</button>` : ""}</p>`);
  }
  els.docList.innerHTML = html.join("");
  syncSelection();
}

/** Checkboxes, the live count and the action bar follow the selection. */
function syncSelection() {
  const sel = lib.selection;
  els.docList.querySelectorAll("[data-group-check]").forEach((box) => UI.setCheck(box, sel.groupState(box.dataset.groupCheck)));
  els.docList.querySelectorAll("[data-file-check]").forEach((box) => {
    box.checked = sel.has(box.dataset.fileCheck);
    box.closest(".tree-file").classList.toggle("selected", box.checked);
  });
  UI.roll($("#selCount"), sel.size);
  $("#selectBar").classList.toggle("show", sel.size > 0);
  const unsearchable = selectedFiles().filter((f) => !f.searchable).length;
  $("#selHint").textContent = unsearchable ? `其中 ${unsearchable} 个还不能提问` : "";
}

function selectedFiles() {
  const ids = new Set(lib.selection.ids());
  return lib.files.filter((f) => ids.has(f.file_id));
}

/** Run `fn` over ids a few at a time; returns how many succeeded and the first error. */
async function eachLimited(ids, fn, limit = 6) {
  let ok = 0;
  let error = null;
  for (let i = 0; i < ids.length; i += limit) {
    const results = await Promise.allSettled(ids.slice(i, i + limit).map(fn));
    for (const r of results) {
      if (r.status === "fulfilled") ok++;
      else error ||= r.reason;
    }
  }
  return { ok, error };
}

function renderRoots() {
  const list = $("#rootList");
  if (!lib.roots.length) {
    list.innerHTML = `<li class="roots-empty">还没有文件夹。${TAURI?.dialog ? "点右上角“添加文件夹”。" : ""}</li>`;
    return;
  }
  list.innerHTML = lib.roots.map((r) => {
    const c = r.counts || {};
    const named = c.named || 0;
    // Files kept by name only are not "to be read": they do not count against 已入库.
    const total = r.total - (c.unsupported || 0) - named;
    const busy = (c.pending || 0) + (c.indexing || 0);
    const parts = total || !named ? [`${fmtNum(c.indexed)} / ${fmtNum(total)} 已入库`] : [];
    if (named) parts.push(r.content_mode === "name" && !total ? "全部只按文件名" : `${fmtNum(named)} 个只按文件名`);
    const excluded = (r.rules || []).filter((rule) => rule.action === "exclude").length;
    if (excluded) parts.push(`${excluded} 处不收录`);
    if (busy) parts.push(`${fmtNum(busy)} 处理中`);
    if (c.missing) parts.push(`<em>${fmtNum(c.missing)} 已丢失</em>`);
    if (c.offline) parts.push(`<em>${fmtNum(c.offline)} 离线</em>`);
    if (c.failed) parts.push(`<em>${fmtNum(c.failed)} 失败</em>`);
    if (r.scanning) parts.unshift("正在扫描…");
    if (c.recycled) parts.push(`<em>${fmtNum(c.recycled)} 在回收站</em>`);
    if (r.status === "offline") parts.unshift("<em>磁盘未连接</em>");
    if (r.status === "missing") parts.unshift("<em>文件夹不存在</em>");
    if (r.status === "recycled") parts.unshift("<em>文件夹在回收站，还原后会自动回来</em>");
    const name = r.path.split(/[\\/]/).filter(Boolean).pop() || r.path;
    const pct = total ? (c.indexed || 0) / total : 0;
    const ring = busy ? `<svg class="ring" viewBox="0 0 24 24" role="img" aria-label="已入库 ${Math.round(pct * 100)}%"><circle class="bg" cx="12" cy="12" r="9"/><circle class="fg" cx="12" cy="12" r="9" stroke-dasharray="56.5" stroke-dashoffset="${(56.5 * (1 - pct)).toFixed(1)}"/></svg>` : "";
    return `<li class="row with-icon root-row" data-root="${esc(r.root_id)}">
      <span class="row-icon${r.status === "active" ? "" : " gray"}">${ICONS.folder}</span>
      <div class="row-main"><strong>${esc(name)}</strong><span class="root-path" title="${esc(r.path)}">${esc(r.path)}</span><span class="root-counts">${parts.join(" · ")}</span></div>
      ${ring}
      ${keepButton(`data-root-keep="${esc(r.root_id)}"`, r.content_mode === "name" || named > 0 || excluded > 0)}
      <button class="icon-btn" type="button" data-rescan="${esc(r.root_id)}" title="重新扫描" aria-label="重新扫描">${ICONS.refresh}</button>
      <button class="icon-btn danger" type="button" data-unroot="${esc(r.root_id)}" title="移除这个文件夹" aria-label="移除这个文件夹">${ICONS.trash}</button>
    </li>`;
  }).join("");
}

function renderProgress(st) {
  const box = $("#progress");
  box.hidden = !st.busy;
  if (!st.busy) return;
  const b = st.batch || {};
  const pct = b.total ? Math.round((b.finished / b.total) * 100) : 0;
  $("#progressBar").style.width = `${pct}%`;
  const reading = AppStatus.jobLines(st.jobs || (st.current ? [st.current] : []));
  $("#progressText").textContent = st.paused
    ? "先回答你的问题，答完接着读"
    : reading.length > 1 ? `${PET}同时在读 ${reading.length} 个文件`
    : reading.length ? `${PET}正在读：${reading[0].name}`
    : st.scanning?.length ? `${PET}正在清点文件夹…` : `${PET}正在读…`;
  $("#progressCount").textContent = b.total ? `${b.finished} / ${b.total}${b.failed ? ` · ${b.failed} 失败` : ""}` : "";
  // Each file being read, one line; a scan shows how far its pages have come.
  const lines = reading.length > 1 || reading[0]?.detail ? reading : [];
  $("#progressJobs").innerHTML = lines.map((job) => `<li>
      <span class="pj-name" title="${esc(job.name)}">${esc(job.name)}</span>
      ${job.detail ? `<span class="pj-detail">${esc(job.detail)}</span><span class="pj-bar"><i style="width:${Math.round(job.progress * 100)}%"></i></span>` : ""}
    </li>`).join("");
}

function renderFilterCounts(counts) {
  const n = (k) => counts[k] || 0;
  const values = {
    all: Object.values(counts).reduce((a, v) => a + v, 0),
    indexed: n("indexed") + n("named"),
    busy: n("pending") + n("indexing"),
    lost: n("missing") + n("offline") + n("recycled"),
    failed: n("failed"),
    unsupported: n("unsupported"),
  };
  document.querySelectorAll("#fileFilters [data-count]").forEach((el) => { el.textContent = values[el.dataset.count] || ""; });
}

/** Poll indexing progress while the library page is open: fast while busy, slow otherwise. */
async function pollStatus() {
  clearTimeout(lib.pollTimer);
  let busy = false;
  try {
    const st = await libraryStatus();
    const changed = lib.status && JSON.stringify(st.counts) !== JSON.stringify(lib.status.counts);
    lib.status = st;
    const rootsBefore = lib.roots.map((r) => r.root_id + r.path + r.content_mode).join("|");
    lib.roots = st.roots || [];
    busy = st.busy;
    renderRoots();
    // The file list groups files under their watched folder: redraw it once the folders are known.
    if (lib.files.length && rootsBefore !== lib.roots.map((r) => r.root_id + r.path + r.content_mode).join("|")) renderFiles(lib.filesTotal);
    renderProgress(st);
    renderFilterCounts(st.counts || {});
    renderStatus();
    if (lib.lostShown !== lostTotal(st)) loadLost();
    if (changed) lib.filesStale = true;
    if (LibraryUI.refreshDue({ stale: lib.filesStale, busy, rows: lib.files.length, sinceLoad: Date.now() - lib.filesLoadedAt })) {
      loadFiles();
      api("/documents").then((d) => { state.docs = d.documents || []; els.docCount.textContent = state.docs.length; }).catch(() => {});
    }
  } catch (_) {}
  if (document.querySelector("#view-library.active")) lib.pollTimer = setTimeout(pollStatus, busy ? 1200 : 5000);
}

/** /library/status plus what changed since the last look; news becomes the status row's note. */
async function libraryStatus() {
  const st = await api(`/library/status?since=${lib.changeSeq}`);
  noteChanges(st.changes, st.busy);
  return st;
}

function noteChanges(changes, busy) {
  if (!changes) return;
  if (changes.seq < changes.since) {
    // The backend restarted and counts from zero again.
    lib.changeSeq = 0;
    return;
  }
  lib.changeSeq = Math.max(lib.changeSeq, changes.seq);
  const note = LibraryUI.changeNote(changes.counts, busy);
  if (note) showNote(note);
  if (note?.tone === "warn" || changes.counts?.restored) lib.lostShown = -1;
}

/** A passing line in the status row (changes the library noticed); it fades after a while. */
function showNote(note, ms = 9000) {
  state.note = { ...note, until: Date.now() + ms };
  renderStatus();
  clearTimeout(showNote.timer);
  showNote.timer = setTimeout(renderStatus, ms + 50);
}

function lostTotal(st) {
  const lost = st?.lost || {};
  return (lost.missing || 0) + (lost.offline || 0) + (lost.recycled || 0);
}

/** "找不到的文件": missing, on a disconnected drive, or in the Recycle Bin. */
async function loadLost() {
  const total = lostTotal(lib.status);
  lib.lostShown = total;
  const group = $("#lostGroup");
  if (!total) { group.hidden = true; return; }
  try {
    const files = (await api("/library/lost?limit=50")).files || [];
    group.hidden = !files.length;
    $("#lostCount").textContent = `${fmtNum(total)} 个`;
    $("#lostList").innerHTML = files.map((f) => {
      const ext = f.name.includes(".") ? `.${f.name.split(".").pop()}` : "";
      const action = f.status === "missing"
        ? `<button class="btn small" type="button" data-locate="${esc(f.file_id)}">重新定位</button>`
        : `<span class="state ${esc(f.status)}">${esc(STATUS_LABEL[f.status])}</span>`;
      return `<li class="row lost-row" data-id="${esc(f.file_id)}" title="${esc(f.path)}">
        ${FileRows.icon(ext)}
        <div class="row-main"><strong>${esc(f.name)}</strong><small>${esc(LibraryUI.lostLine(f))}</small></div>
        ${action}
        <button class="icon-btn sm danger" type="button" data-del="${esc(f.file_id)}" data-name="${esc(f.name)}" title="从知识库移除" aria-label="从知识库移除">${ICONS.trash}</button>
      </li>`;
    }).join("") + (total > files.length
      ? `<li class="lost-more"><button class="btn plain small" type="button" data-lost-all>还有 ${fmtNum(total - files.length)} 个，在文件列表里看</button></li>`
      : "");
  } catch (_) {
    // Not ready yet; the next poll tries again.
    lib.lostShown = -1;
  }
}

/* ---------------- adding files and folders ---------------- */

/** Desktop: native pickers (several at once). Browser: files are uploaded as copies, a
    folder is typed into the row under 文件夹. */
async function pickFiles() {
  if (!TAURI?.dialog) { els.fileInput.click(); return; }
  const picked = await TAURI.dialog.open({
    multiple: true, directory: false, title: `选择要给${PET}读的文件`,
    filters: [{ name: "文档和图片", extensions: LibraryUI.EXTENSIONS }],
  });
  const paths = [].concat(picked || []);
  return paths.length ? importPaths(paths) : false;
}

async function pickFolders() {
  if (!TAURI?.dialog) {
    const path = $("#rootPath").value.trim();
    if (path) importPaths([path]);
    else $("#rootPath").focus();
    return;
  }
  const picked = await TAURI.dialog.open({ directory: true, multiple: true, title: `选择要给${PET}读的文件夹` });
  const paths = [].concat(picked || []);
  return paths.length ? importPaths(paths) : false;
}

/** How to read this batch's images (once per batch), or null when the user backs out. */
async function askImageMode(count, estimate) {
  const s = settingsState.data || (await api("/settings").catch(() => null)) || {};
  const runtime = s.ocr_runtime || {};
  const choice = await UI.chooseSheet({
    title: `怎么读这 ${fmtNum(count)} 张图片？`,
    options: LibraryUI.imageOptions(estimate, {
      ocrInstalled: runtime.installed !== false,
      ocrMB: Math.round((runtime.total_bytes || 0) / 1e6),
      keyConfigured: s.api_key?.configured !== false,
    }),
    value: "local",
    remember: "记住我的选择",
    confirm: "开始读",
  });
  if (choice?.value === "local" && runtime.installed === false) installOcr();
  return choice;
}

/** Add files and folders by path (picked, dropped or typed): look first, ask about images
    once, confirm a very large batch, then add. Nothing is copied. Resolves true when
    something was added. */
async function importPaths(paths) {
  if (lib.importing) { toast(`${PET}还在清点上一批，稍等一下`); return false; }
  lib.importing = true;
  const slow = setTimeout(() => toast(`${PET}正在清点…`), 600);
  try {
    const info = await postJson("/library/inspect", { paths });
    clearTimeout(slow);
    const plan = LibraryUI.plan(info);
    if (!plan.add.length) {
      toast(LibraryUI.importSummary({ results: [...plan.already.map((i) => ({ ...i, ok: true, already: true }))] }, plan.errors).text);
      return false;
    }
    let mode = null;
    let remember = false;
    if (plan.askImages) {
      const choice = await askImageMode(plan.images, info.estimate?.images);
      if (!choice) return false;
      ({ value: mode, remember } = choice);
    }
    if (plan.large && !await UI.confirmSheet({
      title: `添加 ${fmtNum(plan.files)} 个文件？`,
      message: `全部读完${LibraryUI.duration(plan.seconds)}。读完的部分马上就能提问。`,
      confirm: "添加",
    })) return false;
    const result = await postJson("/library/import", { paths: plan.add.map((i) => i.path), image_mode: mode, remember });
    if (remember && settingsState.data) { settingsState.data.image_import_mode = mode; renderSettings(); }
    const summary = LibraryUI.importSummary(result, plan.errors);
    toast(summary.ok ? `${summary.text}，${PET}开始读了` : summary.text);
    $("#rootPath").value = "";
    lib.lostShown = -1;
    if (document.querySelector("#view-library.active")) { await pollStatus(); loadFiles(); }
    else watchLibrary();
    loadDocs();
    return summary.ok;
  } catch (err) {
    clearTimeout(slow);
    toast(`添加失败：${err.message || err}`);
    return false;
  } finally {
    lib.importing = false;
  }
}

async function removeRoot(id) {
  const r = lib.roots.find((x) => x.root_id === id);
  const ok = await UI.confirmSheet({
    title: `移除「${(r?.path || id).split(/[\\/]/).filter(Boolean).pop()}」？`,
    message: `${r?.path || ""}\n其中 ${fmtNum(r?.total ?? 0)} 个文件的索引会一并移除，原文件不会被删除。`,
    confirm: "移除",
    danger: true,
  });
  if (!ok) return;
  try {
    await api(`/watch-roots/${encodeURIComponent(id)}`, { method: "DELETE" });
    toast("已移除这个文件夹");
    await loadDocs();
  } catch (err) {
    toast(`移除失败：${err.message || err}`);
  }
}

async function rescanRoot(id, btn) {
  try {
    await UI.runDone(btn, () => api(`/watch-roots/${encodeURIComponent(id)}/rescan`, { method: "POST" }));
    toast("正在重新扫描");
    pollStatus();
  } catch (err) {
    toast(`扫描失败：${err.message || err}`);
  }
}

async function deleteDoc(id, name) {
  const ok = await UI.confirmSheet({ title: `从知识库移除「${name}」？`, message: "只删除索引，原文件不会被删除。", confirm: "移除", danger: true });
  if (!ok) return;
  try {
    await api(`/files/${encodeURIComponent(id)}`, { method: "DELETE" });
    if (viewer.doc?.doc_id === id) closeViewer();
    toast(`已移除 ${name}`);
    await loadDocs();
  } catch (err) {
    toast(`移除失败：${err.message || err}`);
  }
}

async function relocateFile(id) {
  const f = lib.files.find((x) => x.file_id === id) || (viewer.doc?.doc_id === id ? viewer.doc : null);
  const path = TAURI?.dialog
    ? await TAURI.dialog.open({ title: `原文件已丢失，请选择它现在的位置：${f?.name || ""}`, multiple: false, directory: false })
    : prompt(`原文件已丢失：\n${f?.path || ""}\n\n请输入文件现在的完整路径：`, f?.path || "");
  if (!path) return;
  try {
    const updated = await postJson(`/files/${encodeURIComponent(id)}/relocate`, { path });
    toast(updated.status === "indexed" ? "已重新定位，内容未变化，无需重新入库" : "已重新定位，内容有变化，正在重新入库");
    await loadDocs();
    if (viewer.doc?.doc_id === id) openDocument(id, "", { reload: true });
  } catch (err) {
    toast(`重新定位失败：${err.message || err}`);
  }
}

async function revealFile(id) {
  try {
    await api(`/files/${encodeURIComponent(id)}/reveal`, { method: "POST" });
  } catch (err) {
    toast(`无法打开：${err.message || err}`);
  }
}

/* ---------------- what may be read: 只按文件名 and 不收录 (privacy.js) ---------------- */

/** The shield button on files and folders; `on` when something there is kept back. */
function keepButton(attr, on, size = "") {
  return `<button class="icon-btn keep-btn${size ? ` ${size}` : ""}${on ? " on" : ""}" type="button" ${attr} title="收录方式" aria-label="收录方式">${ICONS.shield}</button>`;
}

const KEEP_DONE = {
  full: "会读取内容，读完就能提问",
  name: "已改为只按文件名，内容不再读取",
  exclude: "已不再收录",
  include: "会重新收录",
};

/** Ask how a file or folder is kept, then confirm plainly what that changes (counted by a dry
    run: how many files, how many have an index to drop) before doing it. */
async function changeKeep({ kind, name, current, canExclude, request }) {
  const choice = await UI.chooseSheet({
    title: `「${name}」怎么收录？`,
    options: PrivacyUI.options({ kind, canExclude }),
    value: current,
    confirm: "继续",
  });
  if (!choice || choice.value === current) return false;
  try {
    const impact = await request(choice.value, true);
    if (!(await UI.confirmSheet(PrivacyUI.confirmText(choice.value, { kind, name, impact })))) return false;
    await request(choice.value, false);
    toast(KEEP_DONE[choice.value]);
  } catch (err) {
    toast(`没能改：${err.message || err}`);
    return false;
  }
  await afterKeepChange();
  return true;
}

async function afterKeepChange() {
  lib.lostShown = -1;
  await loadDocs();
  if (document.body.classList.contains("settings-open")) loadPrivacy();
}

function keepFile(id) {
  const f = lib.files.find((x) => x.file_id === id);
  if (!f) return;
  return changeKeep({
    kind: "file",
    name: f.name,
    current: f.content_mode === "name" ? "name" : "full",
    // A file in a watched folder can be left out for good; a single file is just removed.
    canExclude: !!f.root_id,
    request: (action, dry) => action === "exclude"
      ? postJson("/library/rules", { action: "exclude", path: f.path, dry_run: dry })
      : postJson(`/files/${encodeURIComponent(id)}/content-mode`, { mode: action, dry_run: dry }),
  });
}

function keepFolder(group) {
  if (!group?.root) return;
  const isRoot = group.path.toLowerCase() === group.root.path.toLowerCase();
  return changeKeep({
    kind: "folder",
    name: group.name,
    current: PrivacyUI.groupMode(group.files),
    canExclude: !isRoot,
    request: (action, dry) => action === "exclude"
      ? postJson("/library/rules", { action: "exclude", path: group.path, dry_run: dry })
      : postJson("/library/folder-mode", { path: group.path, mode: action, dry_run: dry }),
  });
}

function keepRoot(id) {
  const r = lib.roots.find((x) => x.root_id === id);
  if (!r) return;
  const current = r.content_mode === "name" ? "name" : r.counts?.named ? "mixed" : "full";
  return changeKeep({
    kind: "folder",
    name: PrivacyUI.base(r.path),
    current,
    canExclude: false,
    request: (action, dry) => postJson("/library/folder-mode", { path: r.path, mode: action, dry_run: dry }),
  });
}

const privacyState = { rules: [] };

/** 设置 › 数据去向 › 不上传的文件: how many files are kept by name only, and every rule. */
async function loadPrivacy() {
  try {
    const data = await api("/library/privacy");
    privacyState.rules = PrivacyUI.sortRules(data.rules);
    const named = data.named_files || 0;
    $("#privacyNamedCount").textContent = named ? `${fmtNum(named)} 个文件` : "无";
    $("#privacyRules").innerHTML = privacyState.rules.map((rule) => {
      const row = PrivacyUI.ruleRow(rule);
      return `<li class="row with-icon" data-rule="${esc(rule.rule_id)}">
        <span class="row-icon gray">${Icons.svg(row.icon)}</span>
        <div class="row-main"><strong title="${esc(rule.path || rule.value)}">${esc(row.label)}</strong><small>${esc(row.line)}</small></div>
        <button type="button" class="btn plain small" data-unrule>取消</button>
      </li>`;
    }).join("");
  } catch (_) {
    // No key yet or the library is not ready: the section stays as it is.
  }
}

async function removePrivacyRule(id) {
  const rule = privacyState.rules.find((item) => item.rule_id === id);
  if (!rule) return;
  const url = `/library/rules/${encodeURIComponent(id)}`;
  try {
    const impact = await api(`${url}?dry_run=true`, { method: "DELETE" });
    const kind = rule.kind === "pattern" ? "pattern" : "folder";
    const name = rule.kind === "pattern" ? rule.value : PrivacyUI.ruleRow(rule).label;
    const action = PrivacyUI.undoAction(rule);
    if (!(await UI.confirmSheet(PrivacyUI.confirmText(action, { kind, name, impact })))) return;
    await api(url, { method: "DELETE" });
    toast(KEEP_DONE[action]);
  } catch (err) {
    toast(`没能取消：${err.message || err}`);
    return;
  }
  await afterKeepChange();
}

async function addPrivacyPattern(e) {
  e.preventDefault();
  const input = $("#privacyPattern");
  const pattern = input.value.trim();
  if (!pattern) { input.focus(); return; }
  const action = document.querySelector('input[name="privacyAction"]:checked')?.value || "exclude";
  try {
    const impact = await postJson("/library/rules", { action, pattern, dry_run: true });
    if (!(await UI.confirmSheet(PrivacyUI.confirmText(action, { kind: "pattern", name: pattern, impact })))) return;
    await postJson("/library/rules", { action, pattern });
    input.value = "";
    toast(KEEP_DONE[action]);
  } catch (err) {
    toast(`没能添加：${err.message || err}`);
    return;
  }
  await afterKeepChange();
}

async function uploadFiles(files) {
  const allowed = LibraryUI.EXTENSIONS.map((e) => `.${e}`);
  let mode = null;
  const images = files.filter((f) => LibraryUI.isImage(f.name)).length;
  const preference = settingsState.data?.image_import_mode;
  if (images && LibraryUI.IMAGE_MODES.includes(preference)) mode = preference;
  else if (images) {
    const choice = await askImageMode(images, {});
    if (!choice) return;
    mode = choice.value;
    if (choice.remember) await saveSettings({ image_import_mode: mode }, "记住了图片的识别方式").catch(() => {});
  }
  for (const file of files) {
    const li = document.createElement("li");
    li.innerHTML = `<span class="spinner"></span><span class="name">${esc(file.name)}</span><span class="state">上传中…</span>`;
    els.uploads.prepend(li);
    const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
    try {
      if (!allowed.includes(ext)) throw new Error("不支持的文件类型");
      if (file.size > 20 * 1024 * 1024) throw new Error("超过 20 MB 限制");
      const form = new FormData();
      form.append("files", file);
      if (mode && LibraryUI.isImage(file.name)) form.append("image_mode", mode);
      const data = await api("/documents", { method: "POST", body: form });
      const r = (data.results || [])[0] || {};
      li.className = "ok";
      li.querySelector(".spinner").outerHTML = ICONS.check;
      li.querySelector(".state").textContent = r.status === "pending" ? "已导入，后台入库中" : STATUS_LABEL[r.status] || "完成";
    } catch (err) {
      li.className = "fail";
      li.querySelector(".spinner").outerHTML = ICONS.alert;
      li.querySelector(".state").textContent = err.message || String(err);
    }
  }
  loadDocs();
}

/* ---------------- document viewer ---------------- */

const viewer = {
  el: $("#viewer"),
  body: $("#viewerBody"),
  search: $("#viewerSearch"),
  count: $("#searchCount"),
  doc: null,
  tab: "units",
  query: "",
  token: 0,
  mode: "original", // "original" (Preview, preview.js) or "parsed" (units / parents)
  target: { locator: "", quote: "" },
  parsedShown: false,
};

function locatorLabel(locator) {
  if (!locator) return "";
  const page = locator.match(/^page:(\d+)(?:-(\d+))?$/);
  if (page) return page[2] && page[2] !== page[1] ? `第 ${page[1]}–${page[2]} 页` : `第 ${page[1]} 页`;
  const slide = locator.match(/^slide:(\d+)(?:-(\d+))?$/);
  if (slide) return slide[2] && slide[2] !== slide[1] ? `第 ${slide[1]}–${slide[2]} 张` : `第 ${slide[1]} 张`;
  const rows = locator.match(/rows:(\d+)-(\d+)/);
  if (rows) {
    const sheet = locator.match(/sheet:([^;]+)/);
    const where = rows[1] === rows[2] ? `第 ${rows[1]} 行` : `第 ${rows[1]}–${rows[2]} 行`;
    return sheet ? `${sheet[1].trim()} · ${where}` : where;
  }
  return locator;
}

/** Escape text and wrap every case-insensitive match of the active query. */
function highlight(text) {
  const q = viewer.query;
  if (!q) return esc(text);
  const lower = text.toLowerCase();
  const needle = q.toLowerCase();
  let out = "";
  let from = 0;
  let at;
  while ((at = lower.indexOf(needle, from)) >= 0) {
    out += esc(text.slice(from, at)) + `<mark>${esc(text.slice(at, at + q.length))}</mark>`;
    from = at + q.length;
  }
  return out + esc(text.slice(from));
}

function countMatches(text) {
  if (!viewer.query) return 0;
  const lower = text.toLowerCase();
  const needle = viewer.query.toLowerCase();
  let n = 0;
  let at = 0;
  while ((at = lower.indexOf(needle, at)) >= 0) { n++; at += needle.length; }
  return n;
}

/** Spreadsheet rows are parsed as "第2行 | 列: 值 | 列: 值"; render them as a table. */
function sheetTable(text) {
  const lines = text.split("\n").filter((l) => l.trim());
  const rows = [];
  const columns = [];
  for (const line of lines) {
    const m = line.match(/^第\s*(\d+)\s*行\s*\|\s*(.*)$/);
    if (!m) return null;
    const cells = {};
    for (const part of m[2].split(" | ")) {
      const cut = part.indexOf(": ");
      if (cut < 0) return null;
      const key = part.slice(0, cut).trim();
      if (!columns.includes(key)) columns.push(key);
      cells[key] = part.slice(cut + 2).trim();
    }
    rows.push({ n: m[1], cells });
  }
  if (!rows.length) return null;
  return `<div class="sheet-wrap"><table class="sheet">
    <thead><tr><th>行</th>${columns.map((c) => `<th>${highlight(c)}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr><td>${r.n}</td>${columns.map((c) => `<td>${highlight(r.cells[c] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody>
  </table></div>`;
}

/** First and last page of a "page:N" / "page:N-M" locator, or null. */
function pageRange(locator) {
  const m = (locator || "").match(/^page:(\d+)(?:-(\d+))?$/);
  return m ? [Number(m[1]), Number(m[2] || m[1])] : null;
}

/** Page of character `offset` in a unit, from its [offset, page] map. */
function pageAt(pageMap, offset) {
  let page = pageMap[0][1];
  for (const [start, p] of pageMap) {
    if (start > offset) break;
    page = p;
  }
  return page;
}

/** A unit's lines as <p>, tagged with the pages they cover; multi-page units get page markers. */
function unitParagraphs(u) {
  const map = u.page_map && u.page_map.length ? u.page_map : null;
  const title = (u.heading || "").split(/\s*>\s*/).pop().replace(/\s+/g, "");
  let html = "";
  let shownPage = null;
  let first = true;
  for (const m of u.text.matchAll(/[^\n]+/g)) {
    const line = m[0];
    if (!line.trim()) continue;
    // The section title is already the unit's heading.
    if (first && title && line.replace(/\s+/g, "") === title) { first = false; continue; }
    first = false;
    let attrs = "";
    if (map) {
      const from = pageAt(map, m.index);
      const to = pageAt(map, m.index + line.length - 1);
      attrs = ` data-first="${from}" data-last="${to}"`;
      if (shownPage !== null && from !== shownPage) html += `<span class="pg">第 ${from} 页</span>`;
      shownPage = to;
    }
    const code = /^\s{2,}\S/.test(line) ? ' class="code"' : /^〔(图片说明|图中文字|公式|第 \d+ 页 识别文字)〕/.test(line) ? ' class="figure"' : "";
    html += `<p${attrs}${code}>${highlight(line)}</p>`;
  }
  return html;
}

function headingHtml(heading, withPath = true) {
  if (!heading) return "（文档开头）";
  const parts = heading.split(/\s*>\s*/);
  const last = parts.pop();
  const path = withPath && parts.length ? `<small>${highlight(parts.join(" › "))}</small>` : "";
  return path + highlight(last);
}

function renderViewer() {
  const doc = viewer.doc;
  if (!doc) return;
  document.querySelectorAll(".viewer .tab[data-tab]").forEach((t) => t.classList.toggle("active", t.dataset.tab === viewer.tab));
  let matches = 0;
  let html = "";

  if (viewer.tab === "units") {
    const units = doc.units.map((u, i) => ({ ...u, i, hits: countMatches(u.heading + "\n" + u.text) }));
    const visible = viewer.query ? units.filter((u) => u.hits) : units;
    matches = units.reduce((n, u) => n + u.hits, 0);
    const imageOnly = doc.image_only_pages || [];
    if (imageOnly.length && !viewer.query) {
      html += `<p class="viewer-note">第 ${imageOnly.join("、")} 页只有图片或图形、没有文字层，未能提取内容（暂不支持 OCR）。</p>`;
    }
    html += visible.map((u) => {
      const table = doc.file_type === ".xlsx" ? sheetTable(u.text) : null;
      const paragraphs = unitParagraphs(u);
      const loc = locatorLabel(u.locator) === u.heading ? "" : locatorLabel(u.locator);
      return `<section class="unit" data-unit="${u.i}">
        <div class="unit-head"><h3>${headingHtml(u.heading)}</h3>${loc ? `<span class="loc">${esc(loc)}</span>` : ""}</div>
        ${table || `<div class="unit-text">${paragraphs}</div>`}
      </section>`;
    }).join("");
  } else {
    const parents = doc.parents.map((p) => ({ ...p, hits: countMatches(p.heading + "\n" + p.text) }));
    const visible = viewer.query ? parents.filter((p) => p.hits) : parents;
    matches = parents.reduce((n, p) => n + p.hits, 0);
    const note = `<p class="parents-note">入库时文档被切成 ${doc.parents.length} 个父块。父块再细分为子块写入向量库用于召回；命中子块后，生成回答使用的是下面这些完整父块。点击可展开全文。</p>`;
    html = note + visible.map((p) => `<article class="parent${viewer.query ? " open" : ""}">
        <div class="parent-head"><code>${esc(p.parent_id.split("-").pop())}</code><strong>${headingHtml(p.heading, false)}</strong><span>${pageRange(p.locator) ? `${esc(locatorLabel(p.locator))} · ` : ""}${p.text.length} 字</span></div>
        <div class="parent-text">${highlight(p.text)}</div>
      </article>`).join("");
  }

  if ((viewer.query && !matches) || !html) {
    html = `<div class="viewer-empty">${viewer.query ? `没有找到「${esc(viewer.query)}」` : "这个文档没有可显示的内容。"}</div>`;
  }
  viewer.body.innerHTML = html;
  viewer.count.textContent = viewer.query ? `${matches} 处` : "";
}

function setViewerHeader(source, meta) {
  const ext = source.includes(".") ? `.${source.split(".").pop().toLowerCase()}` : "";
  const badge = $("#viewerExt");
  badge.outerHTML = FileRows.icon(ext).replace('<span class="', '<span id="viewerExt" class="');
  $("#viewerExt").hidden = !ext;
  $("#viewerTitle").textContent = source;
  $("#viewerTitle").title = source;
  $("#viewerMeta").textContent = meta;
}

/** Open a document in the reader. `focus` is a locator ("page:12", "sheet:X;rows:2-9"…);
    `opts.quote` is text to highlight there; `opts.mode` forces "original" or "parsed". */
async function openDocument(docId, focus = "", opts = {}) {
  const target = { locator: focus || "", quote: opts.quote || "" };
  // Already open: move to the new place instead of loading it again.
  if (viewer.doc?.doc_id === docId && document.body.classList.contains("viewing") && !opts.reload && !opts.mode) {
    viewer.target = target;
    if (viewer.mode === "original" && Preview.focus(docId, target)) return;
    if (viewer.mode === "parsed") { if (focus) focusUnit(focus); return; }
  }
  const token = ++viewer.token;
  Preview.close();
  viewer.doc = null;
  viewer.tab = "units";
  viewer.query = "";
  viewer.target = target;
  viewer.parsedShown = false;
  viewer.search.value = "";
  viewer.count.textContent = "";
  const known = state.docs.find((d) => d.doc_id === docId);
  setViewerHeader(known?.source || "", "正在读取…");
  $("#viewerDownload").hidden = true;
  $("#viewerReveal").hidden = true;
  $("#viewerAlert").hidden = true;
  $("#parentCount").textContent = "";
  document.querySelectorAll(".viewer .tab[data-tab]").forEach((t) => t.classList.toggle("active", t.dataset.tab === "units"));
  viewer.el.dataset.mode = "parsed";
  showFetching(viewer.body);
  if (opts.from && !document.body.classList.contains("viewing")) {
    // Opened from a card: the panel grows out of a circle where it was clicked.
    viewer.el.classList.add("revealing");
    document.body.classList.add("viewing");
    UI.revealFrom(viewer.el, opts.from);
    requestAnimationFrame(() => requestAnimationFrame(() => viewer.el.classList.remove("revealing")));
  }
  document.body.classList.add("viewing");
  viewer.el.setAttribute("aria-hidden", "false");

  $("#viewerMode").hidden = true;
  try {
    // Moved, recycled or edited since the list was drawn: the check follows it first.
    await postJson(`/files/${encodeURIComponent(docId)}/check`, {}).catch(() => null);
    if (token !== viewer.token) return;
    const res = await apiFetch(`${API}/documents/${encodeURIComponent(docId)}`);
    if (!res.ok) throw new Error(await errorDetail(res));
    const data = await res.json();
    if (token !== viewer.token) return;
    viewer.doc = data;
    const ext = data.file_type.replace(".", "").toUpperCase();
    setViewerHeader(
      data.source,
      `${ext} · ${data.units.length} 个段落 · ${data.parents.length} 个父块 · ${data.characters.toLocaleString()} 字`,
    );
    $("#parentCount").textContent = data.parents.length;
    const dl = $("#viewerDownload");
    dl.hidden = !data.has_file;
    dl.href = `${API}/documents/${encodeURIComponent(docId)}/file`;
    $("#viewerReveal").hidden = !data.path;
    $("#viewerReveal").title = data.has_file ? `在资源管理器中打开\n${data.path}` : "打开原文件所在的文件夹";
    const alert = $("#viewerAlert");
    alert.hidden = !LOST.has(data.status) && !data.notice;
    if (LOST.has(data.status)) {
      const why = { offline: "原文件所在的磁盘未连接", recycled: "原文件在回收站，还原后会自动回来" }[data.status] || "原文件已丢失";
      alert.innerHTML = `${ICONS.alert}<span>${why}，下面是入库时保存的文本。<br><small>${esc(data.path || "")}</small></span>
        ${data.status === "missing" ? `<button class="btn ghost" type="button" data-locate="${esc(docId)}">重新定位</button>` : ""}`;
    } else if (data.notice === "scanned_pdf") {
      alert.innerHTML = `${Icons.svg("info")}<span>这份 PDF 是扫描件，要识别文字吗？</span>
        <button class="btn ghost" type="button" data-scan="${esc(docId)}">识别文字</button>`;
    } else if (data.notice === "ocr_unavailable") {
      alert.innerHTML = `${Icons.svg("info")}<span>本机识别组件还没下载，这张图暂时只按文件名入库。</span>
        <button class="btn ghost" type="button" data-ocr-install>下载</button>`;
    }
    const isImage = LibraryUI.isImage(data.source);
    $("#viewerMode").hidden = !isImage || !data.image_mode || LOST.has(data.status);
    if (isImage && data.image_mode) {
      setViewerHeader(data.source, `${ext} · ${IMAGE_MODE_LABEL[data.image_mode] || ""}${data.characters ? ` · ${data.characters.toLocaleString()} 字` : ""}`);
    }
    const parsed = data.parsed !== false;
    if (!parsed) {
      // Registered but never parsed (failed, unsupported, queued): only the original exists.
      const why = { failed: "解析失败", unsupported: "不支持解析", pending: "等待解析", indexing: "正在解析" }[data.status] || "未解析";
      $("#viewerMeta").textContent = `${ext} · ${why}${data.size ? ` · ${fmtSize(data.size)}` : ""}`;
      $("#viewerMeta").title = data.error || "";
    }
    viewer.el.classList.toggle("no-parsed", !parsed);
    $('.viewer .tab[data-mode="parsed"]').disabled = !parsed;
    const canOriginal = (data.preview?.renderer || "parsed") !== "parsed";
    const origTab = $('.viewer .tab[data-mode="original"]');
    origTab.disabled = !canOriginal;
    origTab.title = canOriginal ? "按原文件的样子显示" : "原文件不可读取，只能查看入库时解析的文本";
    setViewerMode(opts.mode && (canOriginal || opts.mode === "parsed") ? opts.mode : canOriginal ? "original" : "parsed");
  } catch (err) {
    if (token !== viewer.token) return;
    $("#viewerMeta").textContent = "";
    viewer.el.dataset.mode = "parsed";
    viewer.body.innerHTML = `<div class="error-box">无法读取文档：${esc(err.message || err)}</div>`;
  }
}

/** Original file (preview.js) or the parsed text; each is built the first time it is shown. */
function setViewerMode(mode) {
  if (!viewer.doc) return;
  viewer.mode = mode;
  viewer.el.dataset.mode = mode;
  document.querySelectorAll(".viewer .tab[data-mode]").forEach((t) => t.classList.toggle("active", t.dataset.mode === mode));
  if (mode === "original") {
    if (!Preview.showing(viewer.doc.doc_id)) Preview.open(viewer.doc, viewer.target);
    return;
  }
  if (!viewer.parsedShown) {
    viewer.parsedShown = true;
    renderViewer();
    if (viewer.target.locator) focusUnit(viewer.target.locator);
  }
}

/** Scroll to the unit a citation points at (matched by heading or page/row label). */
function focusUnit(locator) {
  const want = locator.trim();
  if (!want) return;
  const units = viewer.doc.units;
  let index = units.findIndex((u) => u.heading === want || u.locator === want || locatorLabel(u.locator) === want);
  // A page citation ("page:12-13") inside a section that spans many pages.
  const pages = pageRange(want);
  if (index < 0 && pages) {
    index = units.findIndex((u) => {
      const range = pageRange(u.locator);
      return range && range[0] <= pages[0] && pages[0] <= range[1];
    });
  }
  if (index < 0) {
    index = units.findIndex((u) => u.heading && (u.heading.endsWith(want) || want.endsWith(u.heading.split(/\s*>\s*/).pop())));
  }
  if (index < 0) return;
  const unit = viewer.body.querySelector(`[data-unit="${index}"]`);
  if (!unit) return;
  let el = unit;
  if (pages && units[index].page_map) {
    const lines = [...unit.querySelectorAll("p[data-first]")];
    const line = lines.find((p) => Number(p.dataset.first) <= pages[0] && pages[0] <= Number(p.dataset.last))
      || lines.find((p) => Number(p.dataset.first) >= pages[0]);
    if (line && line !== lines[0]) el = line;
  }
  requestAnimationFrame(() => {
    el.scrollIntoView({ block: el === unit ? "start" : "center", behavior: "smooth" });
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 1800);
  });
}

async function openSource(source, locator, docId, opts = {}) {
  if (docId) { openDocument(docId, locator, opts); return; }
  if (!state.docs.length) await loadDocs();
  const doc = state.docs.find((d) => d.source === source);
  if (!doc) { toast(`知识库中找不到「${source}」，可能已被删除`); return; }
  openDocument(doc.doc_id, locator, opts);
}

function closeViewer() {
  viewer.token++;
  Preview.close();
  viewer.doc = null;
  document.body.classList.remove("viewing");
  viewer.el.setAttribute("aria-hidden", "true");
}

$("#viewerClose").addEventListener("click", closeViewer);
$("#viewerScrim").addEventListener("click", closeViewer);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  // The top-most sheet closes first: Settings sits above the reader.
  if (document.body.classList.contains("settings-open")) closeSettings();
  else if (document.body.classList.contains("viewing")) closeViewer();
});
document.querySelectorAll(".viewer .tab[data-mode]").forEach((t) => t.addEventListener("click", () => {
  if (viewer.doc && !t.disabled) setViewerMode(t.dataset.mode);
}));
$("#viewerOrig").addEventListener("click", async (e) => {
  const act = e.target.closest("[data-orig]")?.dataset.orig;
  if (!act || !viewer.doc) return;
  if (act === "parsed") { setViewerMode("parsed"); return; }
  try {
    await api(`/files/${encodeURIComponent(viewer.doc.doc_id)}/open`, { method: "POST" });
  } catch (err) {
    toast(`无法打开：${err.message || err}`);
  }
});
document.querySelectorAll(".viewer .tab[data-tab]").forEach((t) => t.addEventListener("click", () => {
  if (!viewer.doc) return;
  viewer.tab = t.dataset.tab;
  renderViewer();
  viewer.body.scrollTop = 0;
}));
viewer.search.addEventListener("input", () => {
  clearTimeout(viewer.searchTimer);
  viewer.searchTimer = setTimeout(() => {
    viewer.query = viewer.search.value.trim();
    renderViewer();
  }, 120);
});
viewer.body.addEventListener("click", (e) => {
  const text = e.target.closest(".parent-text");
  if (text && window.getSelection().isCollapsed) text.parentElement.classList.add("open");
});

/* ---------------- settings ---------------- */

const settingsState = { data: null };

async function loadSettings() {
  try {
    settingsState.data = await api("/settings");
    renderSettings();
  } catch (err) {
    $("#settingsError").hidden = false;
    $("#settingsError").textContent = `设置暂不可用：${err.message || err}`;
  }
}

const ANSWER_MODE_DESC = {
  fast: "只核对每处出处，更快、更省。",
  strict: "逐句对照原文，没通过就重新找、重新写。",
};

function renderSettings() {
  const s = settingsState.data;
  if (!s) return;
  $("#settingsError").hidden = true;
  const key = s.api_key;
  $("#keyIcon").className = `row-icon ${key.configured ? "green" : "missing"}`;
  $("#keyState").className = `key-state ${key.configured ? "ok" : "missing"}`;
  $("#keyState").innerHTML = key.configured
    ? `<span>已配置 <code>${esc(key.masked)}</code></span><small>${key.source === "keyring" ? "存在系统凭据管理器" : "来自环境变量"}</small>`
    : `<span>还没有配置，${PET}需要它才能读资料和回答</span>`;
  $("#keyClear").hidden = key.source !== "keyring";
  renderModelWheel(s);
  $("#embeddingModel").textContent = s.embedding_model;
  $("#rerankModel").textContent = s.rerank_model;
  const mode = s.image_understanding || "off";
  document.querySelectorAll('input[name="imageMode"]').forEach((r) => { r.checked = r.value === mode; });
  const runtime = s.ocr_runtime || {};
  const download = runtime.installed ? "" : `；首次使用需下载约 ${Math.round((runtime.total_bytes || 0) / 1e6)} MB`;
  $("#imageModeNote").innerHTML = {
    off: "不处理图片，扫描页里的文字也读不到。",
    local: `在本机识别图中文字，不上传${download}。`,
    cloud: `效果最好，<em>图片和扫描页的截图会上传到硅基流动</em>。`,
  }[mode];
  const importMode = s.image_import_mode || "ask";
  document.querySelectorAll('input[name="imageImportMode"]').forEach((r) => { r.checked = r.value === importMode; });
  const ocrMB = Math.round((runtime.total_bytes || 0) / 1e6);
  $("#imageImportNote").innerHTML = {
    ask: "添加图片时问一次，可以勾选“记住我的选择”。",
    local: `在本机识别图中文字，不上传${runtime.installed ? "" : `；先下载约 ${ocrMB} MB`}。`,
    cloud: "效果最好，<em>图片会上传到硅基流动</em>。",
    name: "只记文件名、文件夹和日期，不识别内容。",
  }[importMode] || "";
  if (!installOcr.running) {
    $("#ocrIcon").className = `row-icon ${runtime.installed ? "green" : "gray"}`;
    $("#ocrState").textContent = runtime.installed ? "已下载，图片在本机识别" : `约 ${ocrMB} MB，只需下载一次`;
    $("#ocrDownload").hidden = !!runtime.installed;
  }
  const formula = s.formula_runtime || {};
  $("#formulaOption").hidden = mode !== "local";
  $("#localFormula").checked = !!s.local_formula;
  $("#formulaHint").textContent = formula.installed ? "离线识别，准确率一般" : `需再下载约 ${Math.round((formula.total_bytes || 0) / 1e6)} MB`;
  const answerMode = s.answer_mode || "strict";
  document.querySelectorAll('input[name="answerMode"]').forEach((r) => { r.checked = r.value === answerMode; });
  $("#answerModeDesc").textContent = ANSWER_MODE_DESC[answerMode];
  $("#followupsEnabled").checked = s.followups_enabled !== false;
  $("#keysTab").hidden = !TAURI?.core;
  UI.segmented($("#settingsTabs"));
  $("#closeToTray").checked = s.close_to_tray !== false;
  $("#shortcutInput").value = s.quick_search_shortcut || "Alt+Space";
  $("#shortcutNote").textContent = "在任何程序里按下，弹出搜索框";
  $("#dataDir").textContent = s.data_dir;
  $("#dataDb").textContent = s.database;
  $("#dataLogs").textContent = s.log_dir;
  $("#appVersion").textContent = s.version;
  Safety.render(s.data_safety);
  renderSuggestionSettings(s.suggestions);
  syncThemeControl();
  UI.syncSegments($("#settingsSheet"));
}

/** The chat model on a wheel: the offered models, the current one, then "其他模型…". */
function renderModelWheel(s) {
  const choices = [...s.llm_model_choices];
  if (!choices.includes(s.llm_model)) choices.unshift(s.llm_model);
  const items = choices.map((m) => {
    const cut = m.indexOf("/");
    return { label: cut > 0 ? m.slice(cut + 1) : m, hint: cut > 0 ? m.slice(0, cut) : "", value: m, title: m };
  });
  items.push({ label: "其他模型…", value: "__custom" });
  const index = choices.indexOf(s.llm_model);
  const wheel = settingsState.wheel;
  if (!wheel) {
    settingsState.wheel = new UI.Wheel($("#modelWheel"), { items, index, label: "问答模型", onChange: pickModel });
  } else if (wheel.items.map((i) => i.value).join() !== items.map((i) => i.value).join()) {
    wheel.setItems(items, index);
  } else if (!wheel.raf && wheel.index !== index) {
    wheel.setIndex(index);
  }
  if (index >= 0) $("#modelCustom").hidden = true;
}

async function pickModel(item) {
  if (item.value === "__custom") {
    $("#modelCustom").hidden = false;
    $("#modelCustomInput").focus();
    return;
  }
  $("#modelCustom").hidden = true;
  try {
    await saveSettings({ llm_model: item.value }, `问答模型换成了 ${item.label}`);
  } catch (err) {
    toast(`切换失败：${err.message || err}`);
    renderSettings();
  }
}

/* ---------------- theme: system, light or dark ---------------- */

function currentTheme() {
  return document.documentElement.dataset.theme || "system";
}

function syncThemeControl() {
  const theme = currentTheme();
  document.querySelectorAll('input[name="themeMode"]').forEach((r) => { r.checked = r.value === theme; });
  UI.segmented($("#themeModes"));
}

/** Switch themes behind a circle growing from where it was asked for. */
function setTheme(theme, source) {
  const root = document.documentElement;
  return UI.circleTransition(source, () => {
    if (theme === "system") delete root.dataset.theme;
    else root.dataset.theme = theme;
    try {
      if (theme === "system") localStorage.removeItem("theme");
      else localStorage.setItem("theme", theme);
    } catch (_) {}
    syncThemeControl();
  });
}

async function saveSettings(body, message) {
  const data = await api("/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  settingsState.data = data;
  renderSettings();
  toast(message);
  checkHealth();
}

$("#keyForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#keyInput");
  const value = input.value.trim();
  try {
    await UI.runDone($("#keySave"), async () => {
      if (!value) throw new Error("先粘贴新的 Key");
      if (value.length < 8 || /\s/.test(value)) throw new Error("这不像一个完整的 Key");
      await saveSettings({ api_key: value }, "Key 已保存到系统凭据管理器");
    });
    input.value = "";
    state.modelIssue = "";
    renderStatus();
  } catch (err) {
    toast(`保存失败：${err.message || err}`);
  }
});
$("#keyTest").addEventListener("click", async (e) => {
  const typed = $("#keyInput").value.trim();
  try {
    const r = await UI.runDone(e.currentTarget, () => postJson("/settings/test-key", typed ? { api_key: typed } : {}));
    const model = r.model_available === false ? `，但这个 Key 用不了 ${r.model}` : "";
    if (!typed && r.model_available !== false) { state.modelIssue = ""; renderStatus(); }
    toast(`${typed ? "这个" : "已保存的"} Key 可以用 · ${r.elapsed_ms} 毫秒${model}`);
  } catch (err) {
    toast(`测试没通过：${err.message || err}`);
  }
});
$("#keyClear").addEventListener("click", async () => {
  if (!await UI.confirmSheet({ title: "删除已保存的 Key？", message: "从系统凭据管理器删除后，需要重新粘贴才能提问。", confirm: "删除", danger: true })) return;
  try {
    settingsState.data = await api("/settings/api-key", { method: "DELETE" });
    renderSettings();
    toast("已删除保存的 API Key");
    checkHealth();
  } catch (err) {
    toast(`删除失败：${err.message || err}`);
  }
});
document.querySelectorAll('input[name="answerMode"]').forEach((radio) => {
  radio.addEventListener("change", async () => {
    $("#answerModeDesc").textContent = ANSWER_MODE_DESC[radio.value];
    try {
      await saveSettings({ answer_mode: radio.value }, radio.value === "fast" ? "换成快速核对" : "换成严谨核对");
    } catch (err) {
      toast(`切换失败：${err.message || err}`);
      renderSettings();
    }
  });
});
$("#followupsEnabled").addEventListener("change", async (e) => {
  const enabled = e.target.checked;
  try {
    await saveSettings({ followups_enabled: enabled }, enabled ? "回答后会推荐追问" : "已关闭回答后推荐追问");
    state.followupsEnabled = enabled;
    if (!enabled) followupRequests.cancel(state.current);
    renderFollowups();
  } catch (err) {
    toast(`设置失败：${err.message || err}`);
    e.target.checked = !enabled;
  }
});
$("#closeToTray").addEventListener("change", async (e) => {
  const enabled = e.target.checked;
  try {
    await applyCloseToTray(enabled);
    await saveSettings({ close_to_tray: enabled }, enabled ? "关闭窗口时将最小化到托盘" : "关闭窗口将直接退出程序");
  } catch (err) {
    toast(`设置失败：${err.message || err}`);
    e.target.checked = !enabled;
  }
});

/** Tell the desktop shell what closing the main window should do. */
function applyCloseToTray(enabled) {
  return TAURI?.core ? TAURI.core.invoke("set_close_to_tray", { enabled }) : Promise.resolve();
}

$("#keyReveal").addEventListener("click", (e) => {
  e.preventDefault();
  const input = $("#keyInput");
  input.type = input.type === "password" ? "text" : "password";
});
/** Download a local recognition component, show progress, then apply the setting that needed it. */
async function downloadOcrRuntime(component = "ocr", body = { image_understanding: "local" }, done = "本地识别组件已就绪，已切换为本地识别") {
  const note = $("#imageModeNote");
  const name = component === "formula" ? "本地公式识别组件" : "本地识别组件";
  const inputs = [...document.querySelectorAll('input[name="imageMode"]'), $("#localFormula")];
  inputs.forEach((r) => { r.disabled = true; });
  try {
    let status = await api(`/ocr-runtime/install?component=${component}`, { method: "POST" });
    while (status.state === "downloading") {
      const pct = status.total_bytes ? Math.min(100, Math.round((status.done_bytes / status.total_bytes) * 100)) : 0;
      note.innerHTML = `正在下载${name}… <b>${pct}%</b>（${(status.done_bytes / 1e6).toFixed(1)} / ${(status.total_bytes / 1e6).toFixed(0)} MB）<span class="dl-bar"><i style="width:${pct}%"></i></span>`;
      await new Promise((r) => setTimeout(r, 1000));
      status = await api(`/ocr-runtime?component=${component}`);
    }
    if (!status.installed) throw new Error(status.error || "下载未完成");
    await saveSettings(body, done);
    const n = settingsState.data.requeued || 0;
    if (n) toast(`${done}，${n} 个 PDF 将在后台重新入库`);
  } catch (err) {
    toast(`下载失败：${err.message || err}`);
    note.innerHTML = `${name}下载失败：${esc(err.message || String(err))}。可以稍后重试。`;
  } finally {
    inputs.forEach((r) => { r.disabled = false; });
    if (settingsState.data) {
      settingsState.data = await api("/settings").catch(() => settingsState.data);
      renderSettings();
    }
  }
}

$("#localFormula").addEventListener("change", async (e) => {
  const on = e.target.checked;
  const runtime = settingsState.data?.formula_runtime || {};
  if (on && !runtime.installed) {
    const size = Math.round((runtime.total_bytes || 0) / 1e6);
    if (!await UI.confirmSheet({ title: `下载公式识别组件（约 ${size} MB）？`, message: "只需下载一次。离线识别准确率一般，和原文对不上的结果会自动丢弃。", confirm: "下载" })) {
      renderSettings();
      return;
    }
    renderSettings();
    downloadOcrRuntime("formula", { local_formula: true }, "已开启本地公式识别");
    return;
  }
  try {
    await saveSettings({ local_formula: on }, on ? "已开启本地公式识别" : "已关闭本地公式识别");
    const n = settingsState.data.requeued || 0;
    if (n) toast(`${n} 个 PDF 将在后台重新入库`);
  } catch (err) {
    toast(`切换失败：${err.message || err}`);
    renderSettings();
  }
});

$("#imageModes").addEventListener("change", async (e) => {
  const mode = e.target.value;
  const current = settingsState.data?.image_understanding || "off";
  if (mode === "cloud" && !settingsState.data?.api_key?.configured) {
    toast("云端多模态模型需要先配置硅基流动 API Key");
    renderSettings();
    return;
  }
  if (mode === "cloud" && !await UI.confirmSheet({ title: "使用云端识别？", message: "PDF 里插图和扫描页的截图会发送到硅基流动。\n有不想外传的资料时，请选“本机”。", confirm: "使用云端" })) {
    renderSettings();
    return;
  }
  if (mode === current) return;
  if (mode === "local" && !settingsState.data?.ocr_runtime?.installed) {
    const runtime = settingsState.data?.ocr_runtime || {};
    const size = Math.round((runtime.total_bytes || 0) / 1e6);
    if (!await UI.confirmSheet({ title: `下载本机识别组件（约 ${size} MB）？`, message: "只需下载一次，下载完自动切换为本机识别。", confirm: "下载" })) {
      renderSettings();
      return;
    }
    renderSettings();
    downloadOcrRuntime();
    return;
  }
  try {
    const label = { off: "已关闭图片理解", local: "已切换为本地识别", cloud: "已切换为云端多模态模型" }[mode];
    await saveSettings({ image_understanding: mode }, label);
    const n = settingsState.data.requeued || 0;
    if (n) toast(`${label}，${n} 个 PDF 将在后台重新入库`);
  } catch (err) {
    toast(`切换失败：${err.message || err}`);
    renderSettings();
  }
});
/** Progress of the local component's download, in its own row. */
function ocrProgress(status) {
  const total = status.total_bytes || 0;
  const pct = total ? Math.min(100, Math.round((status.done_bytes / total) * 100)) : 0;
  $("#ocrDownload").hidden = true;
  $("#ocrState").innerHTML = status.installed
    ? "已下载，图片在本机识别"
    : `正在下载… <b>${pct}%</b>（${(status.done_bytes / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(0)} MB）<span class="dl-bar"><i style="width:${pct}%"></i></span>`;
}

$("#ocrDownload").addEventListener("click", () => {
  installOcr(ocrProgress).then(() => toast("本机识别组件已就绪")).catch(() => renderSettings());
});

$("#imageImportModes").addEventListener("change", async (e) => {
  if (e.target.name !== "imageImportMode") return;
  const mode = e.target.value;
  const s = settingsState.data || {};
  if (mode === "cloud" && !s.api_key?.configured) {
    toast("云端识别需要先配置硅基流动 API Key");
    renderSettings();
    return;
  }
  if (mode === "cloud" && !await UI.confirmSheet({
    title: "以后的图片都用云端识别？",
    message: "图片会发送到硅基流动。\n有不想外传的图片时，请选“本机”或“每次问”。",
    confirm: "使用云端",
  })) {
    renderSettings();
    return;
  }
  const label = { ask: "添加图片时会问一次", local: "以后的图片在本机识别", cloud: "以后的图片用云端识别", name: "以后的图片只记文件名" }[mode];
  try {
    await saveSettings({ image_import_mode: mode }, label);
  } catch (err) {
    toast(`设置失败：${err.message || err}`);
    renderSettings();
    return;
  }
  const runtime = settingsState.data?.ocr_runtime || {};
  if (mode === "local" && !runtime.installed && await UI.confirmSheet({
    title: `下载本机识别组件（约 ${Math.round((runtime.total_bytes || 0) / 1e6)} MB）？`,
    message: "只需下载一次。下载前添加的图片先按文件名入库，下载后自动补读。",
    confirm: "下载",
    cancel: "以后再说",
  })) installOcr(ocrProgress).catch(() => renderSettings());
});

$("#modelCustom").addEventListener("submit", async (e) => {
  e.preventDefault();
  const value = $("#modelCustomInput").value.trim();
  if (!value) return;
  try {
    await saveSettings({ llm_model: value }, `问答模型已切换为 ${value}`);
  } catch (err) {
    toast(`切换失败：${err.message || err}`);
  }
});
$("#openDataDir").addEventListener("click", async () => {
  try {
    await api("/settings/open-data-dir", { method: "POST" });
  } catch (err) {
    toast(`无法打开：${err.message || err}`);
  }
});

/* ---------------- updates (builds with a release host) ---------------- */

async function checkForUpdate() {
  const button = $("#updateCheck");
  const state = $("#updateState");
  button.disabled = true;
  state.textContent = "正在检查…";
  try {
    const update = await TAURI.core.invoke("check_update");
    if (!update) { state.textContent = "已经是最新版本"; return; }
    state.textContent = `有新版本 ${update.version}`;
    const ok = await UI.confirmSheet({
      title: `更新到 ${update.version}？`,
      message: [update.notes, "寻寻会关闭并重新打开。资料和设置不受影响。"].filter(Boolean).join("\n\n"),
      confirm: "更新",
    });
    if (!ok) return;
    state.textContent = "正在下载…";
    await TAURI.core.invoke("install_update");
  } catch (err) {
    state.textContent = "";
    toast(`更新失败：${err.message || err}`);
  } finally {
    button.disabled = false;
  }
}

/* ---------------- guided start ---------------- */

async function startGuide(config) {
  let libraryEmpty = true;
  if (config.api_key.configured) {
    // Only someone with a key can have a library to look at; damaged counts as "has files".
    const status = await api("/library/status").catch(() => null);
    libraryEmpty = !!status && !status.roots?.length && !Object.values(status.counts || {}).some(Boolean);
  }
  const plan = Guide.steps({
    onboardingDone: config.onboarding_done,
    noticeSeen: config.data_notice_seen,
    keyConfigured: config.api_key.configured,
    libraryEmpty,
  });
  if (!plan.length) return;
  Guide.open(plan, {
    esc,
    icon: (name) => Icons.svg(name),
    mascot: () => Icons.mascot(),
    toast,
    desktop: !!TAURI?.dialog,
    currentModel: () => settingsState.data?.llm_model || "",
    remember: (body) => api("/settings", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
    testKey: (key) => postJson("/settings/test-key", { api_key: key }),
    saveKey: async (key) => {
      await saveSettings({ api_key: key }, "Key 已保存");
      state.modelIssue = "";
    },
    saveModel: (model) => saveSettings({ llm_model: model }, `问答模型换成了 ${model.split("/").pop()}`),
    pickFolders,
    pickFiles,
    importPaths,
    onDone: () => {
      checkHealth().then(() => watchLibrary());
      loadDocs();
      loadSuggestions();
      els.input.focus();
    },
  });
}

/* ---------------- health ---------------- */

/** What the status row knows: backend reachable, key set, services ready, chat model. */
const health = { connected: null, keyConfigured: null, ready: null, readyError: "", model: "", damaged: false };

async function checkHealth() {
  try {
    const config = await api("/settings");
    settingsState.data = config;
    health.connected = true;
    health.keyConfigured = !!config.api_key.configured;
    health.damaged = config.data_safety?.state === "damaged";
    // Once per run: the conversations went back to a backup. It stays readable in Settings.
    if (config.data_safety?.notice && !checkHealth.noticed) { checkHealth.noticed = true; toast(config.data_safety.notice); }
    if (!checkHealth.guided) { checkHealth.guided = true; startGuide(config); }
  } catch (err) {
    health.connected = err.message === BACKEND_UNREACHABLE ? false : health.connected;
  }
  if (health.damaged) {
    renderStatus();
    if (!checkHealth.warned) { checkHealth.warned = true; openSettings("about", "#safetyAlert"); }
    return;
  }
  if (health.keyConfigured === false) {
    renderStatus();
    // New installs are asked by the guided start instead.
    if (!checkHealth.prompted) {
      checkHealth.prompted = true;
      if (settingsState.data?.onboarding_done) openSettings("key", "#keyInput");
    }
    return;
  }
  try {
    const res = await apiFetch(`${API}/health/ready`);
    if (!res.ok) throw new Error(await errorDetail(res));
    const data = await res.json();
    health.connected = true;
    health.ready = true;
    health.readyError = "";
    health.model = data.models?.llm || "";
  } catch (err) {
    if (err.message === BACKEND_UNREACHABLE) health.connected = false;
    else health.ready = false;
    health.readyError = err.message || "";
  }
  renderStatus();
}

/** The sidebar's status row: a green dot, or a few words while reading files or when
    something is wrong; clicking goes where it can be dealt with. */
function renderStatus() {
  const note = state.note && state.note.until > Date.now() ? state.note : null;
  const s = AppStatus.describe({ ...health, library: lib.status, modelIssue: state.modelIssue, note });
  const el = els.status;
  el.dataset.tone = s.tone;
  el.dataset.target = s.target || "";
  els.statusText.textContent = s.text;
  els.statusText.hidden = !s.text;
  const bar = $("#statusBar");
  bar.hidden = s.progress == null;
  if (s.progress != null) bar.firstElementChild.style.width = `${Math.round(s.progress * 100)}%`;
  const tip = s.tone === "ok" ? `一切就绪${health.model ? ` · ${health.model}` : ""}` : s.title || s.text;
  el.title = tip;
  el.setAttribute("aria-label", tip || "状态");
  const row = $("#healthText");
  if (row) row.textContent = s.tone === "ok" ? `就绪 · ${health.model}` : s.text || "—";
}

/** Indexing progress for the status row while the library page is not open (it polls itself). */
async function watchLibrary() {
  clearTimeout(watchLibrary.timer);
  if (!document.querySelector("#view-library.active") && health.keyConfigured) {
    try {
      lib.status = await libraryStatus();
    } catch (_) {
      // Not ready yet (still starting, or no key): the health check reports that.
    }
    renderStatus();
  }
  watchLibrary.timer = setTimeout(watchLibrary, lib.status?.busy ? 3000 : 15000);
}

/* ---------------- home: 猜你想问 ---------------- */

const DEFAULT_PLACEHOLDER = "问点什么，或者让寻寻帮你找文件…";
const SUGGEST_N = 6;
const home = { data: null, next: null, prefetching: null, retry: 0, rail: null };

function greeting(now = new Date()) {
  const h = now.getHours();
  if (h >= 5 && h < 11) return "早上好";
  if (h >= 11 && h < 13) return "中午好";
  if (h >= 13 && h < 18) return "下午好";
  if (h >= 18 && h < 23) return "晚上好";
  return "夜深了";
}

function libraryIsEmpty() {
  return !!home.data?.enabled && home.data.ready !== false && !!home.data.library_empty;
}

function renderHomeHead() {
  $("#greeting").textContent = greeting();
  const n = home.data?.file_count || state.docs.length;
  $("#homeSubtitle").textContent = libraryIsEmpty()
    ? "给寻寻一些资料吧，文件留在原处，只记位置。"
    : n ? `我读过你的 ${fmtNum(n)} 个文件，想查什么尽管问。` : "想查什么尽管问，我去你的资料里找。";
}

/* The empty home has one focal point: a greeting and the input with 寻寻 perched on it. The
   input stays one element, pinned to the bottom of the chat view; on the home it is lifted
   onto a reserved slot, and when a conversation starts the lift goes and it glides down. */

function setHome(show) {
  if (show === !els.empty.hidden) { syncHome(); return; }
  if (!show) fadeOutHome();
  els.empty.hidden = !show;
  syncHome();
}

/** A conversation is opening: the home fades over it instead of vanishing. */
function fadeOutHome() {
  if (UI.reduced() || !document.querySelector("#view-chat.active")) return;
  const view = $("#view-chat");
  const from = els.empty.getBoundingClientRect();
  const at = view.getBoundingClientRect();
  if (!from.height) return;
  const ghost = els.empty.cloneNode(true);
  ghost.removeAttribute("id");
  ghost.querySelectorAll("[id]").forEach((n) => n.removeAttribute("id"));
  ghost.setAttribute("aria-hidden", "true");
  ghost.classList.add("home-ghost");
  Object.assign(ghost.style, {
    left: `${from.left - at.left}px`, top: `${from.top - at.top}px`, width: `${from.width}px`, height: `${from.height}px`,
  });
  view.append(ghost);
  ghost.addEventListener("animationend", () => ghost.remove());
  setTimeout(() => ghost.remove(), 600);
}

function syncHome() {
  const showing = !els.empty.hidden;
  const mode = AppStatus.homeMode({ conversation: !showing, libraryEmpty: libraryIsEmpty() });
  document.body.classList.toggle("home", showing && !!document.querySelector("#view-chat.active"));
  els.empty.classList.toggle("is-start", mode === "start");
  $("#homeStart").hidden = mode !== "start";
  const lifted = mode === "focal";
  if (lifted !== els.composer.classList.contains("at-home")) {
    // Only this change glides (not the first layout); following a resize or a scroll must not lag.
    if (syncHome.painted) {
      els.composer.classList.add("moving");
      clearTimeout(syncHome.settle);
      syncHome.settle = setTimeout(() => els.composer.classList.remove("moving"), 700);
    }
    els.composer.classList.toggle("at-home", lifted);
  }
  placeComposer();
}

/** Lift the input onto the home's slot; the slot keeps room for it (and a scope chip above it). */
function placeComposer() {
  if (!els.composer.classList.contains("at-home")) {
    els.composer.style.transform = "";
    return;
  }
  const view = $("#view-chat");
  if (!view.classList.contains("active")) return;
  const dock = $("#composerDock");
  const bar = $("#scopeBar");
  const slot = $("#homeSlot");
  slot.style.height = `${dock.offsetHeight + (bar.hidden ? 0 : bar.offsetHeight + 8)}px`;
  const slotBottom = slot.getBoundingClientRect().bottom - view.getBoundingClientRect().top;
  const naturalBottom = els.composer.offsetTop + dock.offsetTop + dock.offsetHeight;
  els.composer.style.transform = `translateY(${Math.round(slotBottom - naturalBottom)}px)`;
}

/** A little life while someone is at the input: a hop on focus, ears up while typing. */
function petReact(kind) {
  const pet = $("#composerPet .mascot");
  if (!pet || !els.composer.classList.contains("at-home")) return;
  pet.classList.remove("happy");
  if (kind === "focus") {
    void pet.getBoundingClientRect();
    pet.classList.add("happy");
    return;
  }
  pet.classList.add("working");
  clearTimeout(petReact.timer);
  petReact.timer = setTimeout(() => pet.classList.remove("working"), 700);
}

function updatePlaceholder() {
  const conv = currentConversation();
  const ghost = ghostFollowup();
  els.composer.classList.toggle("ghosting", !!ghost);
  $("#ghostKeys").hidden = !ghost;
  if (ghost) {
    // The grey text cannot carry the chip's tag: say when sending leaves the file.
    els.input.placeholder = ghost.scope_action === "clear" ? `在整个知识库中问：${ghost.text}` : ghost.text;
    return;
  }
  els.input.placeholder = conv && conv.messages.length ? "继续追问，或换个问题…" : DEFAULT_PLACEHOLDER;
}

function suggestionCard(c) {
  const tag = AppStatus.cardTag(c);
  const foot = tag ? `<b class="sg-tag">${esc(tag)}</b>` : Icons.svg(c.qtype === "find" ? "folder" : "doc");
  const file = `<span class="sg-file">${foot}<span>${esc(c.file_name)}</span></span>`;
  return `<div class="sg-card" role="listitem" data-id="${esc(c.id)}" data-q="${esc(c.question)}" data-doc="${esc(c.doc_id)}" data-kind="${esc(c.kind)}">
      <button type="button" class="sg-ask" title="${esc(c.question)}&#10;${esc(c.file_name)}"><span class="sg-q">${esc(c.question)}</span>${file}</button>
      <button type="button" class="sg-more" aria-label="更多操作" title="更多操作">${ICONS.more}</button>
      <div class="sg-menu" hidden>
        <button type="button" data-act="dismiss">不感兴趣</button>
        <button type="button" data-act="block">不再推荐这个文件</button>
      </div>
    </div>`;
}

function shownIds() {
  return [...document.querySelectorAll("#suggestions .sg-card")].map((n) => n.dataset.id);
}

function sendSuggestionEvent(action, ids) {
  if (ids.length) postJson("/suggestions/events", { action, ids }).catch(() => {});
}

function closeSuggestionMenus() {
  document.querySelectorAll(".sg-menu").forEach((m) => { m.hidden = true; });
  document.querySelectorAll(".sg-card.menu-open").forEach((c) => c.classList.remove("menu-open"));
}

function applySuggestions(data) {
  home.data = data;
  const enabled = !!data && data.enabled && data.ready !== false;
  const cards = enabled ? data.cards || [] : [];
  $("#homeInfo").hidden = !enabled;
  if (enabled) {
    $("#homeInfoText").textContent = (data.personalize === "history"
      ? "推荐问题来自你的资料和你在本机的提问，只存在这台电脑上。"
      : "推荐问题来自你的资料，只存在这台电脑上。")
      + (data.ai_questions ? "AI 写问题已开启：文件名和开头一小段会发给模型服务。" : "");
  }
  $("#suggestHead").hidden = !cards.length;
  $("#suggestRail").hidden = !cards.length;
  const track = $("#suggestions");
  track.innerHTML = cards.map(suggestionCard).join("");
  track.scrollLeft = 0;
  home.rail ||= new UI.SnapRail($("#suggestRail"));
  home.rail.active = 0;
  requestAnimationFrame(() => home.rail.draw(false));
  if (!UI.reduced()) {
    track.querySelectorAll(".sg-card").forEach((c, i) => c.animate?.(
      [{ opacity: 0, transform: "translateX(20px)" }, { opacity: 1, transform: "none" }],
      { duration: 420, delay: i * 40, easing: "cubic-bezier(.22, 1, .36, 1)", fill: "backwards" },
    ));
  }
  renderHomeHead();
  updatePlaceholder();
  syncHome();
  if (cards.length && !els.empty.hidden) sendSuggestionEvent("shown", cards.map((c) => c.id));
  // First launch: the pool is still being built in the background; look again shortly.
  clearTimeout(home.retry);
  if (enabled && data.building && cards.length < 4) {
    home.retry = setTimeout(() => { if (!els.empty.hidden) loadSuggestions(); }, 5000);
  }
}

async function fetchSuggestions(exclude = []) {
  const q = new URLSearchParams({ n: String(SUGGEST_N) });
  if (exclude.length) q.set("exclude", exclude.join(","));
  return api(`/suggestions?${q}`);
}

/** The next batch is fetched while this one is on screen, so "换一批" swaps at once. */
function prefetchSuggestions() {
  home.next = null;
  const exclude = (home.data?.cards || []).map((c) => c.id);
  const pending = fetchSuggestions(exclude)
    .then((d) => { if (home.prefetching === pending) home.next = d; })
    .catch(() => {});
  home.prefetching = pending;
}

async function loadSuggestions() {
  try {
    applySuggestions(await fetchSuggestions());
    if (home.data?.cards?.length) prefetchSuggestions();
  } catch (_) {
    applySuggestions(null);
  }
}

function showNextSuggestions() {
  if (home.next?.cards?.length) {
    applySuggestions(home.next);
    prefetchSuggestions();
  } else {
    loadSuggestions();
  }
}

async function shuffleSuggestions() {
  const btn = $("#suggestShuffle");
  btn.classList.remove("spinning");
  void btn.offsetWidth;
  btn.classList.add("spinning");
  document.querySelectorAll("#suggestions .sg-card").forEach((c, i) => setTimeout(() => c.classList.add("leaving"), i * 30));
  await new Promise((r) => setTimeout(r, UI.reduced() ? 0 : 200));
  showNextSuggestions();
}

/** "不感兴趣" hides this question; "不再推荐这个文件" hides every question from its file. */
async function suggestionFeedback(card, act) {
  const doc = card.dataset.doc;
  const gone = act === "block"
    ? [...document.querySelectorAll(`#suggestions .sg-card[data-doc="${CSS.escape(doc)}"]`)]
    : [card];
  try {
    if (act === "block") await postJson("/suggestions/blocked", { doc_id: doc });
    else await postJson("/suggestions/events", { action: "dismiss", ids: [card.dataset.id] });
  } catch (err) {
    toast(`操作失败：${err.message || err}`);
    return;
  }
  toast(act === "block" ? "这个文件不会再出现在推荐里，可在设置中恢复" : "好的，不再推荐这个问题");
  gone.forEach((n) => n.classList.add("leaving"));
  const keep = shownIds().filter((id) => !gone.some((n) => n.dataset.id === id));
  try {
    const data = await fetchSuggestions(keep);
    const kept = (home.data?.cards || []).filter((c) => keep.includes(c.id));
    const fill = (data.cards || []).filter((c) => !kept.some((k) => k.doc_id === c.doc_id)).slice(0, SUGGEST_N - kept.length);
    applySuggestions({ ...data, cards: [...kept, ...fill] });
    prefetchSuggestions();
  } catch (_) {
    gone.forEach((n) => n.remove());
  }
}

/* ---------------- settings: 首页推荐 ---------------- */

function suggestionWords(text) {
  return text.split(/[,，、;；\n]+/).map((w) => w.trim()).filter(Boolean);
}

function renderSuggestionSettings(sg) {
  if (!sg) return;
  $("#sgEnabled").checked = !!sg.enabled;
  document.querySelectorAll('input[name="sgPersonalize"]').forEach((r) => {
    r.checked = r.value === sg.personalize;
    r.disabled = !sg.enabled;
  });
  $("#sgAi").checked = !!sg.ai_questions;
  $("#sgAi").disabled = !sg.enabled;
  const words = sg.sensitive_words || [];
  if (document.activeElement !== $("#sgWords")) $("#sgWords").value = words.join("，");
  $("#sgWordsCount").textContent = `${words.length} 个`;
  const folders = sg.excluded_folders || [];
  $("#sgFoldersCount").textContent = folders.length ? `${folders.length} 个` : "无";
  $("#sgFolders").innerHTML = folders.map((f) => `<li data-folder="${esc(f)}">${ICONS.folder}<code title="${esc(f)}">${esc(f)}</code>
      <button type="button" class="btn plain danger small" data-remove-folder>移除</button></li>`).join("");
  const blocked = sg.blocked_files || [];
  $("#sgBlockedBlock").hidden = !blocked.length;
  $("#sgBlockedCount").textContent = `${blocked.length} 个`;
  $("#sgBlocked").innerHTML = blocked.map((f) => `<li data-doc="${esc(f.doc_id)}"><span title="${esc(f.name)}">${esc(f.name)}</span>
      <button type="button" class="btn plain small" data-unblock>恢复推荐</button></li>`).join("");
  const count = sg.history_count || 0;
  const limit = sg.history_limit || { count: 200, days: 90 };
  const recording = sg.personalize === "history" && sg.enabled;
  $("#sgHistoryNote").textContent = recording
    ? `本机存了 ${fmtNum(count)} 条，最多 ${limit.count} 条、${limit.days} 天`
    : count ? `不再记录，已存的 ${fmtNum(count)} 条会自动过期` : "不记录";
  $("#sgPersonalizeDesc").textContent = sg.personalize === "history" ? "偶尔出一张“接着上次的话题”。" : "不记录你的提问。";
  $("#sgClearHistory").disabled = !count;
  const pool = sg.pool || {};
  $("#sgPoolNote").textContent = pool.verified != null
    ? `已备好 ${fmtNum(pool.verified)} 个问题，来自 ${fmtNum(pool.files)} 个文件${sg.building ? "，正在为新文件准备…" : ""}`
    : "";
}

async function saveSuggestionSettings(body, message) {
  try {
    await saveSettings(body, message);
    loadSuggestions();
  } catch (err) {
    toast(`保存失败：${err.message || err}`);
    renderSettings();
  }
}

$("#sgEnabled").addEventListener("change", (e) => {
  saveSuggestionSettings({ suggestions_enabled: e.target.checked },
    e.target.checked ? "首页会显示推荐问题" : "已关闭首页推荐，也不再记录提问");
});
document.querySelectorAll('input[name="sgPersonalize"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    saveSuggestionSettings({ suggestions_personalize: radio.value },
      radio.value === "history" ? "推荐会同时参考你在本机的提问" : "推荐只根据知识库内容，不再记录提问");
  });
});
$("#sgAi").addEventListener("change", async (e) => {
  if (e.target.checked && !await UI.confirmSheet({
    title: "用 AI 写推荐问题？",
    message: "每个文件的文件名、目录和开头一小段会发给硅基流动，用来写更自然的问题。敏感文件不会发送。",
    confirm: "开启",
  })) {
    e.target.checked = false;
    return;
  }
  saveSuggestionSettings({ suggestions_ai: e.target.checked },
    e.target.checked ? "AI 写问题已开启，会在后台逐个文件准备" : "AI 写问题已关闭");
});
$("#sgWordsSave").addEventListener("click", (e) => {
  UI.runDone(e.currentTarget, () => saveSettings({ suggestions_sensitive_words: suggestionWords($("#sgWords").value) }, "敏感词已保存"))
    .then(() => loadSuggestions())
    .catch((err) => toast(`保存失败：${err.message || err}`));
});
$("#sgWordsReset").addEventListener("click", () => {
  const words = settingsState.data?.suggestions?.default_sensitive_words || [];
  $("#sgWords").value = words.join("，");
  saveSuggestionSettings({ suggestions_sensitive_words: words }, "已恢复默认敏感词");
});
$("#sgFolderForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  let path = $("#sgFolderInput").value.trim();
  if (!path && TAURI?.dialog) path = await TAURI.dialog.open({ directory: true, multiple: false, title: "选择不参与推荐的文件夹" });
  if (!path) return;
  const folders = [...(settingsState.data?.suggestions?.excluded_folders || [])];
  if (!folders.includes(path)) folders.push(path);
  $("#sgFolderInput").value = "";
  saveSuggestionSettings({ suggestions_excluded_folders: folders }, "这个文件夹里的文件不再出现在推荐里");
});
$("#sgFolders").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-folder]");
  if (!li || !e.target.closest("[data-remove-folder]")) return;
  const folders = (settingsState.data?.suggestions?.excluded_folders || []).filter((f) => f !== li.dataset.folder);
  saveSuggestionSettings({ suggestions_excluded_folders: folders }, "已移除");
});
$("#sgBlocked").addEventListener("click", async (e) => {
  const li = e.target.closest("li[data-doc]");
  if (!li || !e.target.closest("[data-unblock]")) return;
  try {
    settingsState.data.suggestions = await api(`/suggestions/blocked/${encodeURIComponent(li.dataset.doc)}`, { method: "DELETE" });
    renderSuggestionSettings(settingsState.data.suggestions);
    toast("已恢复推荐这个文件");
  } catch (err) {
    toast(`操作失败：${err.message || err}`);
  }
});
$("#sgClearHistory").addEventListener("click", async () => {
  if (!await UI.confirmSheet({ title: "清除提问记录？", message: "只清除推荐用的提问记录，对话不受影响。", confirm: "清除", danger: true })) return;
  try {
    const data = await api("/suggestions/history", { method: "DELETE" });
    settingsState.data.suggestions = data;
    renderSuggestionSettings(data);
    toast(`已清除 ${fmtNum(data.removed || 0)} 条提问记录`);
    loadSuggestions();
  } catch (err) {
    toast(`清除失败：${err.message || err}`);
  }
});
if (TAURI?.dialog) {
  $("#sgFolderInput").placeholder = "留空点“添加”可选择文件夹";
}

/* ---------------- views & events ---------------- */

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll(".nav-item").forEach((n) => n.classList.toggle("active", n.dataset.view === name));
  els.viewTitle.textContent = { chat: "问答", library: "知识库" }[name] || "";
  syncHome();
  if (name === "chat") requestAnimationFrame(() => home.rail?.draw(false));
  els.app.classList.remove("nav-open");
  if (name === "library") loadDocs();
  else clearTimeout(lib.pollTimer);
}

/* ---------------- settings: a floating sheet over the page ---------------- */

const SETTINGS_PANES = ["key", "suggest", "image", "keys", "about"];

function setSettingsPane(pane) {
  if (!SETTINGS_PANES.includes(pane) || (pane === "keys" && !TAURI?.core)) pane = "key";
  settingsState.pane = pane;
  document.querySelectorAll('input[name="settingsTab"]').forEach((r) => { r.checked = r.value === pane; });
  document.querySelectorAll("#settingsBody .pane").forEach((p) => { p.hidden = p.dataset.pane !== pane; });
  UI.segmented($("#settingsTabs"));
  $("#settingsBody").scrollTop = 0;
}

/** Open Settings at a pane; `focus` is an element to bring into view (and focus). The page
    behind stays as it is: answers keep streaming. */
function openSettings(pane = settingsState.pane || "key", focus = "") {
  setSettingsPane(pane);
  els.app.classList.remove("nav-open");
  const sheet = $("#settingsSheet");
  if (!document.body.classList.contains("settings-open")) {
    settingsState.opener = document.activeElement;
    document.body.classList.add("settings-open");
    sheet.setAttribute("aria-hidden", "false");
  }
  loadPrivacy();
  loadSettings().then(() => {
    const target = focus && $(focus);
    if (target) {
      target.scrollIntoView({ block: "center", behavior: UI.reduced() ? "auto" : "smooth" });
      target.focus?.({ preventScroll: true });
    } else {
      $("#settingsClose").focus({ preventScroll: true });
    }
  });
}

function closeSettings() {
  if (!document.body.classList.contains("settings-open")) return;
  document.body.classList.remove("settings-open");
  $("#settingsSheet").setAttribute("aria-hidden", "true");
  settingsState.opener?.focus?.({ preventScroll: true });
}

$("#openSettings").addEventListener("click", () => openSettings());
$("#settingsClose").addEventListener("click", closeSettings);
$("#settingsScrim").addEventListener("click", closeSettings);
$("#settingsTabs").addEventListener("change", (e) => {
  if (e.target.name === "settingsTab") setSettingsPane(e.target.value);
});
els.status.addEventListener("click", () => {
  const target = els.status.dataset.target;
  if (target === "library") showView("library");
  else if (target === "lost") {
    showView("library");
    loadLost().then(() => $("#lostGroup").scrollIntoView({ behavior: UI.reduced() ? "auto" : "smooth", block: "center" }));
  }
  else if (target === "key") openSettings("key", "#keyInput");
  else if (target) openSettings(target);
});

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const run = state.running.get(state.current);
  if (run) run.controller.abort();
  else if (!els.input.value.trim() && ghostFollowup()) sendFollowup(ghostFollowup());
  else submitInput();
});
els.input.addEventListener("input", () => {
  autosize();
  // Typing (an input method starting to compose included) hides the suggestion at once and
  // drops a request still on its way; with the input emptied again the suggestion is back.
  if (els.input.value) followupRequests.cancel(state.current);
  else state.filled = null;
  updatePlaceholder();
  if (els.input.value) warmConnections();
});

/* While a question is typed, the backend reopens its idle connections to the model service,
   so sending it does not wait for those first (free requests, at most every 20 s). */
let lastWarm = 0;
function warmConnections() {
  const now = Date.now();
  if (now - lastWarm < 20000) return;
  lastWarm = now;
  apiFetch(`${API}/warm`, { method: "POST" }).catch(() => {});
}
els.input.addEventListener("keydown", (e) => {
  const ghost = ghostFollowup();
  const act = Followups.keyAction(e, { value: els.input.value, active: !!ghost });
  if (act) {
    e.preventDefault();
    if (act === "fill") fillFollowup(ghost);
    else if (act === "send") sendFollowup(ghost);
    else dismissGhost();
    return;
  }
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    if (!isRunning()) submitInput();
  }
});

$("#suggestions").addEventListener("click", (e) => {
  const card = e.target.closest(".sg-card");
  if (!card) return;
  if (e.target.closest(".sg-more")) {
    const menu = card.querySelector(".sg-menu");
    const open = menu.hidden;
    closeSuggestionMenus();
    menu.hidden = !open;
    card.classList.toggle("menu-open", open);
    return;
  }
  const act = e.target.closest(".sg-menu button")?.dataset.act;
  if (act) { closeSuggestionMenus(); suggestionFeedback(card, act); return; }
  if (e.target.closest(".sg-ask")) {
    sendSuggestionEvent("click", [card.dataset.id]);
    ask(card.dataset.q);
  }
});
document.addEventListener("click", (e) => {
  if (!e.target.closest(".sg-more, .sg-menu")) closeSuggestionMenus();
});
$("#suggestShuffle").addEventListener("click", shuffleSuggestions);
$("#railPrev").addEventListener("click", () => home.rail?.scrollBy(-1));
$("#railNext").addEventListener("click", () => home.rail?.scrollBy(1));
$("#homeInfoLink").addEventListener("click", (e) => {
  e.preventDefault();
  openSettings("suggest");
});
$("#homeAddFolder").addEventListener("click", () => {
  showView("library");
  pickFolders().catch((err) => toast(`无法选择文件夹：${err}`));
});
$("#homeAddFiles").addEventListener("click", () => {
  showView("library");
  pickFiles().catch((err) => toast(`无法选择文件：${err}`));
});

/** The answer's text without citation numbers, for the clipboard. */
function answerText(bot) {
  return Citations.strip(bot?.answer || "").replace(/[ \t]+\n/g, "\n").trim();
}

els.thread.addEventListener("click", async (e) => {
  const retry = e.target.closest('[data-act="retry-ask"], [data-act="retry-cancel"], [data-act="retry-settings"]');
  if (retry) {
    if (retry.dataset.act === "retry-ask") retryAsk(retry.dataset.bot);
    else if (retry.dataset.act === "retry-cancel") cancelAutoRetry();
    else openSettings("key");
    return;
  }

  const head = e.target.closest(".trace-head");
  if (head) {
    const open = head.parentElement.classList.toggle("open");
    head.setAttribute("aria-expanded", String(open));
    return;
  }

  const chip = e.target.closest(".fu-chip");
  if (chip) {
    const bot = currentConversation()?.messages.find((m) => m.id === chip.closest(".followups").dataset.bot);
    sendFollowup(bot?.followups?.items?.[Number(chip.dataset.i)]);
    return;
  }

  const more = e.target.closest(".more-sources");
  if (more) { more.nextElementSibling.hidden = !more.nextElementSibling.hidden; return; }

  const toggle = e.target.closest(".deck-toggle");
  if (toggle) {
    const deck = toggle.closest(".sources").querySelector(".deck")._deck;
    deck?.set(!deck.collapsed);
    return;
  }

  const msgEl = e.target.closest(".msg-bot");
  const bot = msgEl && currentConversation()?.messages.find((m) => m.id === msgEl.dataset.bot);
  const act = e.target.closest(".msg-actions [data-act]");
  if (act?.dataset.act === "copy") {
    try {
      await UI.runDone(act, () => UI.copyText(answerText(bot)));
      toast("回答已复制");
    } catch (_) {
      toast("没能写入剪贴板，可以选中文字后按 Ctrl+C");
    }
    return;
  }
  if (act?.dataset.act === "scope-sources") {
    const ids = [...new Set([...msgEl.querySelectorAll(".deck .source[data-doc]")].map((s) => s.dataset.doc).filter(Boolean))];
    if (ids.length) {
      setScope({ doc_ids: ids, label: ids.length === 1 ? msgEl.querySelector(".deck .source strong").textContent : `这次回答的 ${ids.length} 个来源` });
      toast(`之后的问题只在这 ${ids.length} 个文件里找`);
      els.input.focus();
    }
    return;
  }

  const results = e.target.closest(".file-results");
  if (results) {
    const hit = FileRows.hit(e);
    if (!hit?.row) return;
    const { file: id, name, locator } = hit.row.dataset;
    const found = bot?.files?.[hit.index];
    if (hit.act === "read") openDocument(id, hit.locator ?? locator, { quote: hit.quote ?? found?.snippets?.[0]?.text ?? "", from: e });
    else if (hit.act === "reveal") revealFile(id);
    else if (hit.act === "ask") askAboutFile(id, name);
    return;
  }

  const source = e.target.closest(".source[data-source]");
  if (source) {
    const opts = { from: e, ...(source.dataset.quote ? { quote: source.dataset.quote } : {}) };
    openSource(source.dataset.source, source.dataset.locator, source.dataset.doc, opts);
    return;
  }

  const cite = e.target.closest(".cite");
  if (cite) {
    const card = msgEl?.querySelector(`.source[data-n="${cite.dataset.n}"]`);
    if (!card) return;
    if (card.parentElement.hidden) card.parentElement.hidden = false;
    card.closest(".deck")?._deck?.set(false);
    UI.disconnect(msgEl.querySelector(".bot-body"), true);
    setTimeout(() => {
      card.scrollIntoView({ behavior: UI.reduced() ? "auto" : "smooth", block: "nearest" });
      card.classList.add("flash");
      setTimeout(() => card.classList.remove("flash"), 1400);
    }, UI.reduced() ? 0 : 200);
  }
});

// Hovering a citation draws a line to its source; a collapsed deck lifts that card to the top.
els.thread.addEventListener("pointerover", (e) => {
  const cite = e.target.closest(".cite");
  if (!cite || cite === els.thread._cite) return;
  els.thread._cite = cite;
  const body = cite.closest(".bot-body");
  const deck = body?.querySelector(".deck")?._deck;
  const card = body?.querySelector(`.deck .source[data-n="${cite.dataset.n}"]`);
  if (!deck || !card) return;
  cite.classList.add("on");
  card.classList.add("on");
  if (deck.collapsed) deck.peekAt(deck.cards.indexOf(card));
  setTimeout(() => { if (els.thread._cite === cite) UI.connect(body, cite, card); }, deck.collapsed ? 140 : 0);
});
els.thread.addEventListener("pointerout", (e) => {
  const cite = e.target.closest(".cite");
  if (!cite || cite.contains(e.relatedTarget)) return;
  els.thread._cite = null;
  const body = cite.closest(".bot-body");
  cite.classList.remove("on");
  body?.querySelectorAll(".source.on").forEach((c) => c.classList.remove("on"));
  body?.querySelector(".deck")?._deck?.peekAt(-1);
  UI.disconnect(body);
});

els.history.addEventListener("click", (e) => {
  if (e.target.closest('[data-act="reload-history"]')) { loadConversations(); return; }
  const li = e.target.closest("li[data-id]");
  if (!li) return;
  if (e.target.closest(".del")) {
    state.running.get(li.dataset.id)?.controller.abort();
    state.conversations = state.conversations.filter((c) => c.id !== li.dataset.id);
    if (state.current === li.dataset.id) state.current = null;
    if (state.opening === li.dataset.id) state.opening = null;
    conversationSaver.remove(li.dataset.id);
    renderConversation();
    return;
  }
  openConversation(li.dataset.id);
});

// Like re-tapping an iOS tab: 问答 pressed while already on 问答 starts a new conversation.
document.querySelectorAll(".nav-item").forEach((n) => n.addEventListener("click", (e) => {
  const onChat = $("#view-chat").classList.contains("active");
  if (n.dataset.view === "chat" && onChat && state.current) newConversation(e);
  else showView(n.dataset.view);
}));
$("#refreshDocs").addEventListener("click", (e) => {
  UI.runDone(e.currentTarget, async () => { if (!(await loadDocs())) throw new Error("刷新失败"); }).catch(() => {});
});
els.docList.addEventListener("click", (e) => {
  const groupBox = e.target.closest("[data-group-check]");
  if (groupBox) { lib.selection.toggleGroup(groupBox.dataset.groupCheck); syncSelection(); return; }
  const fileBox = e.target.closest("[data-file-check]");
  if (fileBox) { lib.selection.toggle(fileBox.dataset.fileCheck, fileBox.checked); syncSelection(); return; }
  const b = e.target.closest("[data-del]");
  if (b) { deleteDoc(b.dataset.del, b.dataset.name); return; }
  const chip = e.target.closest("[data-card]");
  if (chip) { showCard(chip.dataset.card); return; }
  const loc = e.target.closest("[data-locate]");
  if (loc) { relocateFile(loc.dataset.locate); return; }
  const rev = e.target.closest("[data-reveal]");
  if (rev) { revealFile(rev.dataset.reveal); return; }
  const scan = e.target.closest("[data-scan]");
  if (scan) { askScannedPdf(scan.dataset.scan); return; }
  const keep = e.target.closest("[data-keep]");
  if (keep) { keepFile(keep.dataset.keep); return; }
  const folderKeep = e.target.closest("[data-folder-keep]");
  if (folderKeep) { keepFolder(lib.groups[Number(folderKeep.dataset.folderKeep)]); return; }
  const folder = e.target.closest("[data-toggle]");
  if (folder) {
    const group = lib.groups[Number(folder.dataset.toggle)];
    if (!group) return;
    const closed = !lib.closed.has(group.key);
    if (closed) lib.closed.add(group.key); else lib.closed.delete(group.key);
    folder.parentElement.classList.toggle("closed", closed);
    return;
  }
  const row = e.target.closest(".tree-file[data-id]");
  if (row) openDocument(row.dataset.id, "", { from: e });
});

$("#addFiles").addEventListener("click", () => pickFiles().catch((err) => toast(`无法选择文件：${err}`)));
$("#addFolder").addEventListener("click", () => pickFolders().catch((err) => toast(`无法选择文件夹：${err}`)));
$("#rootForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const path = $("#rootPath").value.trim();
  if (path) importPaths([path]);
  else $("#rootPath").focus();
});
$("#lostList").addEventListener("click", (e) => {
  const loc = e.target.closest("[data-locate]");
  if (loc) { relocateFile(loc.dataset.locate); return; }
  const del = e.target.closest("[data-del]");
  if (del) { deleteDoc(del.dataset.del, del.dataset.name); return; }
  if (e.target.closest("[data-lost-all]")) {
    const radio = document.querySelector('input[name="fileFilter"][value="missing,offline,recycled"]');
    radio.checked = true;
    radio.dispatchEvent(new Event("change", { bubbles: true }));
    UI.syncSegments($("#fileFilters"));
    $("#docList").scrollIntoView({ behavior: UI.reduced() ? "auto" : "smooth", block: "start" });
    return;
  }
  const row = e.target.closest(".lost-row[data-id]");
  if (row) openDocument(row.dataset.id, "", { from: e });
});
$("#privacyRules").addEventListener("click", (e) => {
  const b = e.target.closest("[data-unrule]");
  if (b) removePrivacyRule(b.closest("[data-rule]").dataset.rule);
});
$("#privacyForm").addEventListener("submit", addPrivacyPattern);
$("#rootList").addEventListener("click", (e) => {
  const keep = e.target.closest("[data-root-keep]");
  if (keep) { keepRoot(keep.dataset.rootKeep); return; }
  const rescan = e.target.closest("[data-rescan]");
  if (rescan) { rescanRoot(rescan.dataset.rescan, rescan); return; }
  const un = e.target.closest("[data-unroot]");
  if (un) removeRoot(un.dataset.unroot);
});
$("#fileFilters").addEventListener("change", (e) => {
  if (e.target.name !== "fileFilter") return;
  lib.filter = e.target.value;
  lib.fileLimit = LibraryUI.FILE_PAGE;
  loadFiles();
});
$("#typeFilter").addEventListener("change", (e) => {
  lib.docType = e.target.value;
  lib.fileLimit = LibraryUI.FILE_PAGE;
  loadFiles();
});
let fileSearchTimer = 0;
$("#fileSearch").addEventListener("input", (e) => {
  clearTimeout(fileSearchTimer);
  fileSearchTimer = setTimeout(() => {
    lib.search = e.target.value.trim();
    lib.fileLimit = LibraryUI.FILE_PAGE;
    loadFiles();
  }, 250);
});
els.docList.addEventListener("click", (e) => {
  if (!e.target.closest("#filesMore")) return;
  lib.fileLimit += LibraryUI.FILE_PAGE;
  loadFiles();
});

// Batch actions on the selected files.
$("#selClear").addEventListener("click", () => { lib.selection.clear(); syncSelection(); });
$("#selAsk").addEventListener("click", (e) => {
  const files = selectedFiles().filter((f) => f.searchable);
  if (!files.length) { toast("选中的文件都还没读完，暂时不能提问"); return; }
  const scope = { doc_ids: files.map((f) => f.file_id), label: files.length === 1 ? files[0].name : `选中的 ${files.length} 个文件` };
  UI.circleTransition(e, () => {
    lib.selection.clear();
    syncSelection();
    state.current = null;
    renderConversation();
    showView("chat");
    setScope(scope);
  }).then(() => {
    toast(`之后的问题只在${files.length === 1 ? `「${files[0].name}」` : `这 ${files.length} 个文件`}里找`);
    els.input.focus();
  });
});
$("#selReindex").addEventListener("click", async (e) => {
  const ids = lib.selection.ids();
  try {
    const { ok, error } = await UI.runDone(e.currentTarget, async () => {
      const result = await eachLimited(ids, (id) => api(`/files/${encodeURIComponent(id)}/reindex`, { method: "POST" }));
      if (!result.ok && result.error) throw result.error;
      return result;
    });
    toast(`已把 ${ok} 个文件排进队列，${PET}会重新读一遍${error ? `；有些没成功：${error.message || error}` : ""}`);
    pollStatus();
    loadFiles();
  } catch (err) {
    toast(`重新入库失败：${err.message || err}`);
  }
});
$("#selRemove").addEventListener("click", async (e) => {
  const ids = lib.selection.ids();
  const ok = await UI.confirmSheet({ title: `从知识库移除 ${ids.length} 个文件？`, message: "只删除索引，原文件不会被删除。", confirm: "移除", danger: true });
  if (!ok) return;
  try {
    const result = await UI.runDone(e.currentTarget, async () => {
      const r = await eachLimited(ids, (id) => api(`/files/${encodeURIComponent(id)}`, { method: "DELETE" }));
      if (!r.ok && r.error) throw r.error;
      return r;
    });
    if (viewer.doc && ids.includes(viewer.doc.doc_id)) closeViewer();
    lib.selection.clear();
    toast(`已移除 ${result.ok} 个文件${result.error ? `；有些没成功：${result.error.message || result.error}` : ""}`);
    await loadDocs();
  } catch (err) {
    toast(`移除失败：${err.message || err}`);
  }
});
$("#viewerReveal").addEventListener("click", () => { if (viewer.doc) revealFile(viewer.doc.doc_id); });
$("#viewerDownload").addEventListener("click", async (e) => {
  if (!DESKTOP?.token) return;
  // A plain link cannot carry the token header: fetch the file and save it as a blob.
  e.preventDefault();
  try {
    const res = await apiFetch(e.currentTarget.href);
    if (!res.ok) throw new Error(await errorDetail(res));
    const url = URL.createObjectURL(await res.blob());
    const a = Object.assign(document.createElement("a"), { href: url, download: viewer.doc?.source || "document" });
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  } catch (err) {
    toast(`下载失败：${err.message || err}`);
  }
});
$("#viewerAlert").addEventListener("click", (e) => {
  const b = e.target.closest("[data-locate]");
  if (b) relocateFile(b.dataset.locate);
  const scan = e.target.closest("[data-scan]");
  if (scan) askScannedPdf(scan.dataset.scan);
  if (e.target.closest("[data-ocr-install]")) installOcr();
});
$("#viewerMode").addEventListener("click", () => { if (viewer.doc) changeImageMode(viewer.doc); });

/** Read one file's images another way: the file is read again in the background. */
async function setFileImageMode(id, mode, message) {
  try {
    await postJson(`/files/${encodeURIComponent(id)}/image-mode`, { mode });
    toast(message);
    if (viewer.doc?.doc_id === id) openDocument(id, "", { reload: true });
    pollStatus();
    loadFiles();
  } catch (err) {
    toast(`没能改：${err.message || err}`);
  }
}

/** "这份 PDF 是扫描件，要识别文字吗？" — local or cloud for this PDF only; "不用了" stops asking. */
async function askScannedPdf(id) {
  const s = settingsState.data || (await api("/settings").catch(() => null)) || {};
  const runtime = s.ocr_runtime || {};
  const options = LibraryUI.imageOptions({}, {
    ocrInstalled: runtime.installed !== false,
    ocrMB: Math.round((runtime.total_bytes || 0) / 1e6),
    keyConfigured: s.api_key?.configured !== false,
  }).filter((o) => o.value !== "name").map((o) => ({ ...o, line: o.line.replace("图片会上传", "页面截图会上传") }));
  const choice = await UI.chooseSheet({
    title: "识别这份扫描件的文字？", options, value: "local", confirm: "识别", cancel: "不用了", cancelValue: "off",
  });
  if (!choice) return;
  if (choice.value === "off") { await setFileImageMode(id, "off", "好的，这份 PDF 不再提醒"); return; }
  if (choice.value === "local" && runtime.installed === false) installOcr();
  await setFileImageMode(id, choice.value, `${PET}开始识别这份扫描件`);
}

/** Change how one image file is read (the reader's "换一种方式"). */
async function changeImageMode(doc) {
  const s = settingsState.data || (await api("/settings").catch(() => null)) || {};
  const runtime = s.ocr_runtime || {};
  const options = LibraryUI.imageOptions({}, {
    ocrInstalled: runtime.installed !== false,
    ocrMB: Math.round((runtime.total_bytes || 0) / 1e6),
    keyConfigured: s.api_key?.configured !== false,
  });
  const choice = await UI.chooseSheet({ title: `怎么读「${doc.source}」？`, options, value: doc.image_mode, confirm: "重新读" });
  if (!choice || choice.value === doc.image_mode) return;
  if (choice.value === "local" && runtime.installed === false) installOcr();
  await setFileImageMode(doc.doc_id, choice.value, `${PET}会按${IMAGE_MODE_LABEL[choice.value]}重新读这张图`);
}

/** Download the local recognition component in the background (it changes no setting);
    images waiting for it are read once it is in. `onProgress(status)` follows the download. */
async function installOcr(onProgress = null) {
  if (installOcr.running) return installOcr.running;
  installOcr.running = (async () => {
    try {
      let status = await api("/ocr-runtime/install?component=ocr", { method: "POST" });
      if (!status.installed) toast(`正在下载本机识别组件（约 ${Math.round((status.total_bytes || 0) / 1e6)} MB）`);
      while (status.state === "downloading") {
        onProgress?.(status);
        await new Promise((r) => setTimeout(r, 1000));
        status = await api("/ocr-runtime?component=ocr");
      }
      if (!status.installed) throw new Error(status.error || "下载未完成");
      onProgress?.(status);
      postJson("/library/sync", {}).catch(() => {});
      if (settingsState.data) { settingsState.data.ocr_runtime = status; renderSettings(); }
      return status;
    } catch (err) {
      toast(`本机识别组件下载失败：${err.message || err}`);
      throw err;
    } finally {
      installOcr.running = null;
    }
  })();
  return installOcr.running;
}

els.fileInput.addEventListener("change", () => {
  if (els.fileInput.files.length) uploadFiles([...els.fileInput.files]);
  els.fileInput.value = "";
});
/* Files and folders dropped anywhere on the window. The desktop shell hands over their
   paths, so nothing is copied; a browser only gives the files' contents, which are uploaded
   as copies (folders cannot be dropped there). */
function showDropVeil(on) {
  $("#dropVeil").classList.toggle("show", on);
  $("#dropVeil").setAttribute("aria-hidden", String(!on));
}
const currentWebview = TAURI?.webview?.getCurrentWebview?.();
if (currentWebview) {
  let carrying = false;
  currentWebview.onDragDropEvent(({ payload }) => {
    if (payload.type === "enter") {
      carrying = (payload.paths || []).length > 0;
      showDropVeil(carrying);
    } else if (payload.type === "over") {
      if (carrying) showDropVeil(true);
    } else if (payload.type === "leave") {
      carrying = false;
      showDropVeil(false);
    } else if (payload.type === "drop") {
      carrying = false;
      showDropVeil(false);
      if (payload.paths?.length) importPaths(payload.paths);
    }
  }).catch(() => {});
} else {
  let depth = 0;
  const carriesFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
  document.addEventListener("dragenter", (e) => {
    if (!carriesFiles(e)) return;
    depth++;
    showDropVeil(true);
  });
  document.addEventListener("dragleave", (e) => {
    if (!carriesFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) showDropVeil(false);
  });
  document.addEventListener("dragover", (e) => { if (carriesFiles(e)) e.preventDefault(); });
  document.addEventListener("drop", (e) => {
    if (!carriesFiles(e)) return;
    e.preventDefault();
    depth = 0;
    showDropVeil(false);
    const files = [...(e.dataTransfer?.files || [])];
    if (files.length) uploadFiles(files);
  });
}
["dragenter", "dragover"].forEach((t) => els.dropzone.addEventListener(t, () => els.dropzone.classList.add("over")));
["dragleave", "drop"].forEach((t) => els.dropzone.addEventListener(t, () => els.dropzone.classList.remove("over")));

$("#themeModes").addEventListener("change", (e) => {
  if (e.target.name === "themeMode") setTheme(e.target.value, e.target.closest("label"));
});

$("#menuBtn").addEventListener("click", () => els.app.classList.add("nav-open"));
$("#sidebarClose").addEventListener("click", () => els.app.classList.remove("nav-open"));
$("#scrim").addEventListener("click", () => els.app.classList.remove("nav-open"));

/* ---------------- boot ---------------- */

hydrateIcons();
$("#brandMark").innerHTML = Icons.BRAND;
$("#homePet").innerHTML = Icons.mascot();
$("#composerPet").innerHTML = Icons.mascot();
$("#aboutPet").innerHTML = Icons.mascot();
// Anything that moves the slot moves the input with it: its own height, the window, the
// home's layout settling (under reduced motion the thread's padding still takes a frame).
const homeLayout = new ResizeObserver(() => placeComposer());
[els.composer, els.scroller, els.empty].forEach((el) => homeLayout.observe(el));
els.scroller.addEventListener("scroll", () => {
  if (document.body.classList.contains("home")) placeComposer();
}, { passive: true });
els.input.addEventListener("focus", () => petReact("focus"));
els.input.addEventListener("input", () => petReact("typing"));
UI.largeTitle($("#libraryPage"));
UI.syncSegments();
syncThemeControl();
if (TAURI?.dialog) document.body.classList.add("desktop");
if (TAURI?.opener) {
  // The webview cannot open new tabs: send external links to the default browser.
  document.addEventListener("click", (e) => {
    const a = e.target.closest('a[target="_blank"]');
    if (!a || !/^https:/.test(a.href)) return;
    e.preventDefault();
    TAURI.opener.openUrl(a.href).catch((err) => toast(`无法打开链接：${err}`));
  });
}

/* ---------------- quick search (desktop) ---------------- */

/** "Ctrl+Shift+K" from a keydown; null while only modifiers are held. */
function acceleratorFromEvent(e) {
  const named = { " ": "Space", Enter: "Enter", Tab: "Tab", "`": "Backquote" };
  let key = named[e.key] || (/^F([1-9]|1[0-2])$/.test(e.key) ? e.key : null);
  if (!key && /^Key[A-Z]$/.test(e.code)) key = e.code.slice(3);
  if (!key && /^Digit[0-9]$/.test(e.code)) key = e.code.slice(5);
  const mods = [e.ctrlKey && "Ctrl", e.altKey && "Alt", e.shiftKey && "Shift", e.metaKey && "Super"].filter(Boolean);
  return key && mods.length ? [...mods, key].join("+") : null;
}

async function registerShortcut(accelerator) {
  if (!TAURI?.core) return false;
  await TAURI.core.invoke("set_quick_search_shortcut", { accelerator });
  return true;
}

$("#shortcutInput").addEventListener("keydown", (e) => {
  if (e.key === "Escape" || e.key === "Backspace") return;
  e.preventDefault();
  const accelerator = acceleratorFromEvent(e);
  if (accelerator) e.currentTarget.value = accelerator;
});
$("#shortcutForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const accelerator = $("#shortcutInput").value.trim();
  const previous = settingsState.data?.quick_search_shortcut || "Alt+Space";
  if (!accelerator || accelerator === previous) return;
  const btn = $("#shortcutSave");
  try {
    await UI.runDone(btn, () => registerShortcut(accelerator));
  } catch (err) {
    $("#shortcutInput").value = previous;
    toast(`无法使用 ${accelerator}：${err}（仍为 ${previous}）`);
    return;
  }
  try {
    settingsState.data = await api("/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ quick_search_shortcut: accelerator }),
    });
    renderSettings();
    toast(`快速搜索快捷键改成了 ${accelerator}`);
  } catch (err) {
    toast(`快捷键已生效，但保存失败：${err.message || err}`);
  }
});

if (TAURI?.core) {
  api("/settings")
    .then((s) => {
      applyCloseToTray(s.close_to_tray !== false).catch(() => {});
      return registerShortcut(s.quick_search_shortcut || "Alt+Space");
    })
    .catch((err) => toast(`快速搜索快捷键未能注册：${err.message || err}，可在设置中更换`));
  TAURI.event?.listen("open-reader", ({ payload }) => {
    showView("chat");
    openDocument(payload.file_id, payload.locator || "", { quote: payload.quote || "" });
  });
  TAURI.event?.listen("ask-file", ({ payload }) => {
    showView("chat");
    askAboutFile(payload.file_id, payload.name || "");
  });
}

/* ---------------- scope picker events ---------------- */

$("#scopeBtn").addEventListener("click", openScopePicker);
$("#scopeFilter").addEventListener("input", renderScopeList);
$("#scopeList").addEventListener("click", (e) => {
  const li = e.target.closest("li");
  if (!li) return;
  if (li.hasAttribute("data-scope-all")) setScope(null);
  else if (li.dataset.scopeFolder) setScope({ folder: li.dataset.scopeFolder, label: li.dataset.scopeFolder.split(/[\\/]/).pop() || li.dataset.scopeFolder });
  else if (li.dataset.scopeFile) setScope({ doc_ids: [li.dataset.scopeFile], label: li.dataset.name });
  else return;
  $("#scopePop").hidden = true;
  els.input.focus();
});
$("#scopeBar").addEventListener("click", (e) => {
  if (e.target.closest(".scope-clear")) setScope(null);
});
document.addEventListener("click", (e) => {
  if (!e.target.closest("#scopePop, #scopeBtn")) $("#scopePop").hidden = true;
});

/* The window is back in front: look at the disk now (single files have no watcher), then
   ask for the changes, which the status row turns into a passing note. */
let lastSync = 0;
function syncOnFocus() {
  if (document.hidden || !health.keyConfigured || Date.now() - lastSync < 10000) return;
  lastSync = Date.now();
  postJson("/library/sync", {})
    .then(() => (document.querySelector("#view-library.active") ? pollStatus() : watchLibrary()))
    .catch(() => {});
}
window.addEventListener("focus", syncOnFocus);
document.addEventListener("visibilitychange", syncOnFocus);
$("#dropPet").innerHTML = Icons.mascot();
if (TAURI?.core) {
  TAURI.core.invoke("update_configured").then((configured) => {
    if (!configured) return;
    $("#updateGroup").hidden = false;
    $("#updateCheck").addEventListener("click", checkForUpdate);
  }).catch(() => {});
}
Safety.init({
  esc,
  toast,
  postJson,
  runDone: UI.runDone,
  confirm: UI.confirmSheet,
  onChanged: () => { checkHealth().then(() => watchLibrary()); loadDocs(); },
  restart: TAURI?.core ? () => TAURI.core.invoke("restart_app") : null,
});

renderConversation();
renderHomeHead();
requestAnimationFrame(() => { syncHome.painted = true; });
updateSend();
loadConversations();
checkHealth().then(() => watchLibrary());
// Wakes the backend's idle preparation of recently added Word files.
setTimeout(() => postJson("/documents/prepare", { doc_ids: [] }).catch(() => {}), 5000);
loadDocs();
loadSuggestions();
api("/settings")
  .then((s) => { state.followupsEnabled = s.followups_enabled !== false; renderFollowups(); })
  .catch(() => {});
els.input.focus();
