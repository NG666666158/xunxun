"use strict";

/* Conversations live in the data folder (SQLite, behind /conversations), used by app.js:
   moving what localStorage still holds, and saving each conversation on its own so several
   can stream at once. Kept free of the page so tests can run it under Node. */
(function (root) {
  const LEGACY_KEY = "rag.conversations.v1";
  const MOVED_KEY = "rag.conversations.moved";

  const RETRY_DELAYS = [2000, 5000, 15000, 30000];

  /** Send what localStorage holds to the backend; its own copy goes only once the backend
      has confirmed every conversation. Anything else leaves it for the next launch. */
  async function moveLegacy(storage, send, newId = () => `t-${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`) {
    const raw = storage.getItem(LEGACY_KEY);
    if (!raw) return { moved: 0 };
    if (storage.getItem(MOVED_KEY)) {
      storage.removeItem(LEGACY_KEY);
      return { moved: 0 };
    }
    let saved;
    try {
      saved = JSON.parse(raw);
    } catch (_) {
      return { moved: 0, unreadable: true };
    }
    const conversations = (Array.isArray(saved) ? saved : [])
      .filter((c) => c && typeof c === "object")
      .map((c) => (c.id ? c : { ...c, id: newId() }));
    if (conversations.length) {
      const { ids = [] } = await send(conversations);
      const stored = new Set(ids.map(String));
      const missing = conversations.filter((c) => !stored.has(String(c.id))).length;
      if (missing) throw new Error(`还有 ${missing} 段对话没存进去`);
    }
    storage.setItem(MOVED_KEY, "1");
    storage.removeItem(LEGACY_KEY);
    return { moved: conversations.length };
  }

  /** The list from the backend next to conversations started while it loaded, newest first.
      Those here already may be mid-stream, so they keep their objects. */
  function mergeList(here, fromBackend) {
    const known = new Set(here.map((c) => c.id));
    return [...here, ...fromBackend.filter((c) => !known.has(c.id)).map((c) => ({ ...c, messages: null }))]
      .sort((a, b) => (b.updated || 0) - (a.updated || 0));
  }

  /** A stream cut off by closing the app or a restart reads as interrupted, not as running. */
  function settleMessages(messages, newId) {
    for (const m of messages) {
      if (m.role !== "bot") continue;
      if (!m.id) m.id = newId();
      if (!m.done && !m.error) m.error = "回答被中断了。";
    }
    return messages;
  }

  /** One save at a time per conversation, in order, newest content wins; a failed save is
      retried until it works, and onFailing says while any of them is failing. */
  class Saver {
    constructor({ put, remove, delay = 300, retryDelays = RETRY_DELAYS, onFailing = () => {} }) {
      this.put = put;
      this.removeFn = remove;
      this.delay = delay;
      this.retryDelays = retryDelays;
      this.onFailing = onFailing;
      this.jobs = new Map();
      this.failing = new Set();
    }

    job(id) {
      if (!this.jobs.has(id)) {
        this.jobs.set(id, { id, conv: null, deleted: false, dirty: false, running: false, timer: 0, attempts: 0 });
      }
      return this.jobs.get(id);
    }

    save(conv) {
      const job = this.job(conv.id);
      if (job.deleted) return;
      job.conv = conv;
      job.dirty = true;
      this.schedule(job, this.delay);
    }

    /** Waits for a save in flight, so the delete is the last thing the backend hears. */
    remove(id) {
      const job = this.job(id);
      job.deleted = true;
      job.conv = null;
      job.dirty = true;
      this.schedule(job, 0);
    }

    schedule(job, ms) {
      if (job.running) return;
      clearTimeout(job.timer);
      job.timer = setTimeout(() => this.run(job), ms);
    }

    async run(job, closing = false) {
      if (job.running || !job.dirty) return;
      clearTimeout(job.timer);
      job.running = true;
      job.dirty = false;
      try {
        if (job.deleted) await this.removeFn(job.id, closing);
        else await this.put(job.conv, closing);
        job.attempts = 0;
        this.setFailing(job.id, null);
      } catch (err) {
        job.dirty = true;
        job.attempts += 1;
        this.setFailing(job.id, err);
      } finally {
        job.running = false;
      }
      if (job.dirty) {
        const wait = job.attempts ? this.retryDelays[Math.min(job.attempts, this.retryDelays.length) - 1] : this.delay;
        this.schedule(job, wait);
      } else {
        this.jobs.delete(job.id);
      }
    }

    setFailing(id, err) {
      const was = this.failing.size > 0;
      if (err) this.failing.add(id);
      else this.failing.delete(id);
      const now = this.failing.size > 0;
      if (now !== was || err) this.onFailing(now, err || null);
    }

    /** The page is going away: send everything waiting now. */
    flush() {
      return Promise.all([...this.jobs.values()].map((job) => this.run(job, true)));
    }

    pending() {
      return this.jobs.size;
    }
  }

  root.ConvStore = { LEGACY_KEY, MOVED_KEY, moveLegacy, mergeList, settleMessages, Saver };
})(typeof window !== "undefined" ? window : globalThis);
