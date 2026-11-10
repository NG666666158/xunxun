"use strict";

/* Small decisions behind the UI, kept free of the page so tests can run them under Node:
   what the sidebar's status row says, which files to prepare for preview ahead of time,
   and what 寻寻 says while a file is being fetched. */
(function (root) {
  /** The status row: {tone: "ok" | "busy" | "error" | "wait" | "note" | "warn", text, target, progress}.
      `target` is where a click goes: a Settings pane ("key", "about"), "library" or "lost".
      Problems come first; a quiet green dot means everything is fine. `note` is a passing
      line about changes the library found by itself ({tone, text, target}, see
      LibraryUI.changeNote): files that went missing outrank reading progress, news does not. */
  function describe({ connected, keyConfigured, ready, readyError = "", library = null, modelIssue = "", note = null, damaged = false } = {}) {
    if (connected === false) return { tone: "error", text: "后台没有响应", target: "about" };
    if (damaged) return { tone: "error", text: "索引库已损坏", target: "about" };
    if (keyConfigured === false) return { tone: "error", text: "还没有 Key", target: "key" };
    if (ready === false) return { tone: "error", text: "服务未就绪", target: "about", title: readyError };
    if (modelIssue) return { tone: "error", text: modelIssue, target: "key" };
    if (note?.tone === "warn") return { tone: "warn", text: note.text, target: note.target || "lost" };
    if (library?.busy) {
      const batch = library.batch || {};
      const left = Math.max(0, (batch.total || 0) - (batch.finished || 0));
      const text = note?.text || (left ? `正在读 ${left} 个文件` : "正在清点文件夹");
      const progress = batch.total ? Math.min(1, (batch.finished || 0) / batch.total) : null;
      const reading = jobLines(library.jobs || (library.current ? [library.current] : []));
      const title = library.paused
        ? "先回答你的问题，答完接着读"
        : reading.map((job) => (job.detail ? `${job.name}（${job.detail}）` : job.name)).join("\n");
      return { tone: "busy", text, target: "library", progress, title };
    }
    if (connected == null || ready == null) return { tone: "wait", text: "正在连接…", target: "" };
    if (note?.text) return { tone: "note", text: note.text, target: note.target || "library" };
    return { tone: "ok", text: "", target: "" };
  }

  /** The files being read side by side (/library/status `jobs`), oldest first:
      {name, detail ("16 / 61 页" for a scan), progress (0..1 or null)}. */
  function jobLines(jobs) {
    return (jobs || [])
      .filter((job) => job && job.name)
      .map((job) => {
        const [done, total] = job.pages || [];
        return total
          ? { name: job.name, detail: `${done} / ${total} 页`, progress: Math.min(1, done / total) }
          : { name: job.name, detail: "", progress: null };
      });
  }

  /** How the empty home is laid out: "off" (a conversation is on screen), "start" (nothing
      to ask about yet, so adding files is the one action and the input waits at the bottom),
      "focal" (寻寻, a greeting and the input in the middle). */
  function homeMode({ conversation = false, libraryEmpty = false } = {}) {
    if (conversation) return "off";
    return libraryEmpty ? "start" : "focal";
  }

  /** A suggestion card's small tag: only what the question and file name cannot say. */
  function cardTag(card) {
    return card?.kind === "new" || card?.kind === "continue" ? card.label || "" : "";
  }

  const WORD = /\.(docx|doc)$/i;

  /** Files that are about to be shown get their preview prepared in the background:
      Word files only, each once per session, sent in small batches. */
  class PrepareQueue {
    constructor(send, { delay = 300, batch = 20 } = {}) {
      this.send = send;
      this.delay = delay;
      this.batch = batch;
      this.seen = new Set();
      this.waiting = [];
      this.timer = 0;
    }

    /** items: [{id, name}]; returns how many were newly queued. */
    add(items) {
      let added = 0;
      for (const item of items || []) {
        if (!item?.id || !WORD.test(item.name || "") || this.seen.has(item.id)) continue;
        this.seen.add(item.id);
        this.waiting.push(item.id);
        added++;
      }
      if (added && !this.timer) {
        this.timer = setTimeout(() => this.flush(), this.delay);
      }
      return added;
    }

    flush() {
      clearTimeout(this.timer);
      this.timer = 0;
      while (this.waiting.length) {
        const ids = this.waiting.splice(0, this.batch);
        Promise.resolve(this.send(ids)).catch(() => {
          // The backend was busy or away: these may be asked for again later.
          ids.forEach((id) => this.seen.delete(id));
        });
      }
    }
  }

  /** What 寻寻 says while fetching a file, by how long it has taken so far (ms). */
  function fetchLine(elapsed) {
    if (elapsed < 900) return "寻寻找到文件踪迹";
    if (elapsed < 3500) return "正在把文件叼回来";
    if (elapsed < 9000) return "快到了，再等一下";
    return "这个文件有点大，寻寻还在叼";
  }

  root.AppStatus = { describe, jobLines, homeMode, cardTag, PrepareQueue, fetchLine, WORD };
})(typeof window !== "undefined" ? window : globalThis);
