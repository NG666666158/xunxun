"use strict";

/* When the page looks for a new version by itself (kept free of the page so tests can run it
   under Node). Once a little after each start, then once a day while 寻寻 keeps running in the
   tray; a check that failed (offline, GitHub unreachable) tries again an hour later. Nothing
   is installed from here: a new version only lights a quiet notice, and the user decides. */
(function (root) {
  const FIRST_MS = 30 * 1000; // after the window opens, so the start's own work goes first
  const DAY_MS = 24 * 60 * 60 * 1000;
  const RETRY_MS = 60 * 60 * 1000;
  const TICK_MS = 60 * 1000; // how often the page asks whether a check is due

  /** Whether to check now. Times are Date.now() values (wall clock, so a sleeping PC counts):
      ``startedAt`` the page's start, ``lastTry`` the last check begun (0: none yet),
      ``lastOk`` the last one that got an answer, ``busy`` one is under way. */
  function due({ startedAt, lastTry = 0, lastOk = 0, busy = false }, now) {
    if (busy) return false;
    if (!lastTry) return now - startedAt >= FIRST_MS;
    if (now < lastTry) return true; // the clock went back: the gap means nothing
    return now - lastTry >= (lastOk >= lastTry ? DAY_MS : RETRY_MS);
  }

  /** Ask every minute whether a check is due and run ``check`` (resolves to the new version,
      or null when this one is the newest); ``found`` gets each answer. A failure is only
      logged. Returns ``tick`` (the page also calls it when the window comes back) and the
      state, for tests. */
  function schedule({
    check,
    found,
    log = (message) => console.warn(message),
    now = () => Date.now(),
    every = (fn, ms) => setInterval(fn, ms),
  }) {
    const state = { startedAt: now(), lastTry: 0, lastOk: 0, busy: false };
    async function tick() {
      if (!due(state, now())) return false;
      state.busy = true;
      state.lastTry = now();
      try {
        const update = await check();
        state.lastOk = now();
        found(update || null);
      } catch (err) {
        log(`检查更新失败，稍后再试：${(err && err.message) || err}`);
      } finally {
        state.busy = false;
      }
      return true;
    }
    every(tick, TICK_MS);
    return { tick, state };
  }

  /** Markdown marks a reader of the sheet does not need: links keep their text. */
  const plain = (text) => text
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/(\*\*|__|`)/g, "")
    .trim();

  /** Release notes (a CHANGELOG section) as groups: [{ title, items }]. Headings open a
      group, "- " lines are its items, an indented line after an item joins it, and any other
      text becomes an item of its own, so nothing the release said is dropped. */
  function notes(text) {
    const groups = [];
    let group = null;
    let inItem = false;
    for (const raw of String(text || "").split(/\r?\n/)) {
      const line = raw.trim();
      if (!line) { inItem = false; continue; }
      const heading = line.match(/^#{1,6}\s+(.*)$/);
      if (heading) { group = { title: plain(heading[1]), items: [] }; groups.push(group); inItem = false; continue; }
      if (!group) { group = { title: "", items: [] }; groups.push(group); }
      const bullet = line.match(/^(?:[-*+]|\d+[.)])\s+(.*)$/);
      if (bullet) { group.items.push(plain(bullet[1])); inItem = true; }
      else if (inItem && /^\s/.test(raw)) group.items[group.items.length - 1] += plain(line);
      else { group.items.push(plain(line)); inItem = false; }
    }
    return groups.filter((g) => g.title || g.items.length);
  }

  root.UpdateCheck = { due, schedule, notes, FIRST_MS, DAY_MS, RETRY_MS, TICK_MS };
})(typeof window !== "undefined" ? window : globalThis);
