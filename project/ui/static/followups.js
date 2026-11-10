"use strict";

/* Follow-up suggestions after an answer, used by app.js: which answer may show them, what
   each key does to the grey suggestion in the empty input, and one request per
   conversation that a new question, typing or leaving the conversation cancels.
   Kept free of the page so tests can run it under Node. */
(function (root) {
  /** The last answer of a conversation when its follow-ups may show: finished, not
      running, the conversation's last message, switched on. Earlier answers never do. */
  function host(conv, { running = false, enabled = true } = {}) {
    if (!conv || !enabled || running) return null;
    const last = conv.messages?.[conv.messages.length - 1];
    if (!last || last.role !== "bot" || !last.done || last.error) return null;
    return last.followups?.items?.length ? last : null;
  }

  /** The suggestion shown as grey text: the first one, while the input is empty and it
      has not been dismissed (Esc, or typing in the input). */
  function ghost(conv, opts = {}, value = "") {
    const bot = host(conv, opts);
    return bot && !bot.followups.dismissed && !value ? bot.followups.items[0] : null;
  }

  /** The alternatives below the answer: up to three after the first. */
  function chips(conv, opts = {}) {
    const bot = host(conv, opts);
    return bot ? bot.followups.items.slice(1, 4) : [];
  }

  /** What a keydown in the input does while a suggestion may show: "fill" | "send" | "dismiss" | null. */
  function keyAction(e, { value = "", active = false } = {}) {
    // An input method that is composing owns these keys (Enter commits, Esc cancels it).
    if (!active || value || e.isComposing || e.keyCode === 229) return null;
    if (e.key === "Escape") return "dismiss";
    if (e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) return null;
    if (e.key === "Tab" || e.key === "ArrowRight") return "fill";
    if (e.key === "Enter") return "send";
    return null;
  }

  /** How sending a suggestion changes the conversation's scope: a found file is asked
      about within that file; "在整个知识库中重问" clears the scope. */
  function scopeChange(item) {
    if (item?.scope_action === "set" && item.scope?.doc_ids?.length) {
      return { doc_ids: item.scope.doc_ids, label: item.scope.label || "" };
    }
    if (item?.scope_action === "clear") return null;
    return undefined; // keep
  }

  /** At most one request in flight per key (conversation id). */
  class Requests {
    constructor(fetcher) {
      this.fetcher = fetcher;
      this.live = new Map();
    }

    /** The response of the newest request for key, or null once it was cancelled, replaced or failed. */
    async start(key, body) {
      this.cancel(key);
      const controller = new AbortController();
      this.live.set(key, controller);
      try {
        const data = await this.fetcher(body, controller.signal);
        return controller.signal.aborted ? null : data;
      } catch (_) {
        return null;
      } finally {
        if (this.live.get(key) === controller) this.live.delete(key);
      }
    }

    cancel(key) {
      const controller = this.live.get(key);
      if (!controller) return false;
      controller.abort();
      this.live.delete(key);
      return true;
    }

    pending(key) {
      return this.live.has(key);
    }
  }

  root.Followups = { host, ghost, chips, keyAction, scopeChange, Requests };
})(typeof window !== "undefined" ? window : globalThis);
