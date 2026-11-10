// Conversations in the data folder (project/ui/static/convstore.js): moving localStorage, saving, retrying.
// Run with `node --test project/tests/js/convstore.test.cjs`; test_conversations.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "convstore.js"));
const { LEGACY_KEY, MOVED_KEY, moveLegacy, mergeList, settleMessages, Saver } = globalThis.ConvStore;

function storage(initial = {}) {
  const data = new Map(Object.entries(initial));
  return {
    data,
    getItem: (k) => (data.has(k) ? data.get(k) : null),
    setItem: (k, v) => data.set(k, v),
    removeItem: (k) => data.delete(k),
  };
}
const conv = (id, extra = {}) => ({ id, title: id, updated: 1, messages: [], ...extra });
const tick = (ms = 5) => new Promise((resolve) => setTimeout(resolve, ms));
/** Timers run late on a busy machine: wait for the outcome, not a fixed time. */
async function until(condition, ms = 3000) {
  const end = Date.now() + ms;
  while (!condition()) {
    if (Date.now() > end) assert.fail(`still waiting for: ${condition}`);
    await tick(2);
  }
}
const saver = (put, remove = async () => {}, extra = {}) => new Saver({ put, remove, delay: 0, retryDelays: [1, 1], ...extra });

test("all conversations move, past the old cap of 40, and localStorage lets go only afterwards", async () => {
  const many = Array.from({ length: 95 }, (_, i) => conv(`c${i}`));
  const store = storage({ [LEGACY_KEY]: JSON.stringify(many) });
  let sent = null;
  const result = await moveLegacy(store, async (list) => {
    sent = list;
    assert.ok(store.getItem(LEGACY_KEY), "still there while the backend works");
    return { ids: list.map((c) => c.id) };
  });
  assert.equal(result.moved, 95);
  assert.equal(sent.length, 95);
  assert.equal(store.getItem(LEGACY_KEY), null);
  assert.equal(store.getItem(MOVED_KEY), "1");
});

test("a failed or partial import keeps localStorage for the next launch", async () => {
  const raw = JSON.stringify([conv("a"), conv("b")]);
  const store = storage({ [LEGACY_KEY]: raw });
  await assert.rejects(moveLegacy(store, async () => { throw new Error("backend away"); }), /backend away/);
  assert.equal(store.getItem(LEGACY_KEY), raw);
  assert.equal(store.getItem(MOVED_KEY), null);

  await assert.rejects(moveLegacy(store, async () => ({ ids: ["a"] })), /1 段/);
  assert.equal(store.getItem(LEGACY_KEY), raw);
  assert.equal(store.getItem(MOVED_KEY), null);

  const again = await moveLegacy(store, async (list) => ({ ids: list.map((c) => c.id) }));
  assert.equal(again.moved, 2);
  assert.equal(store.getItem(LEGACY_KEY), null);
});

test("nothing is sent twice, and a copy left behind cannot bring deleted conversations back", async () => {
  const store = storage({ [LEGACY_KEY]: JSON.stringify([conv("a")]), [MOVED_KEY]: "1" });
  const result = await moveLegacy(store, async () => assert.fail("already moved"));
  assert.equal(result.moved, 0);
  assert.equal(store.getItem(LEGACY_KEY), null);
  assert.deepEqual(await moveLegacy(storage(), async () => assert.fail("nothing to move")), { moved: 0 });
});

test("unreadable data is left alone, entries without an id get one", async () => {
  const broken = storage({ [LEGACY_KEY]: "{not json" });
  assert.deepEqual(await moveLegacy(broken, async () => assert.fail("nothing readable")), { moved: 0, unreadable: true });
  assert.equal(broken.getItem(LEGACY_KEY), "{not json");
  assert.equal(broken.getItem(MOVED_KEY), null);

  let sent;
  const store = storage({ [LEGACY_KEY]: JSON.stringify([{ title: "没有 id" }, null, conv("a")]) });
  await moveLegacy(store, async (list) => { sent = list; return { ids: list.map((c) => c.id) }; }, () => "new-id");
  assert.deepEqual(sent.map((c) => c.id), ["new-id", "a"]);
});

test("the list keeps conversations already here, with their objects, newest first", () => {
  const live = conv("live", { updated: 5, messages: [{ role: "user" }] });
  const merged = mergeList([live], [conv("old", { updated: 9 }), conv("live", { updated: 1 }), conv("older", { updated: 2 })]);
  assert.deepEqual(merged.map((c) => c.id), ["old", "live", "older"]);
  assert.equal(merged[1], live);
  assert.equal(merged[0].messages, null, "read when opened");
});

test("an answer that never finished reads as interrupted", () => {
  const ids = ["b1"];
  const messages = settleMessages(
    [{ role: "user", text: "问" }, { role: "bot", done: false, answer: "写到一半" }, { role: "bot", id: "b2", done: true, answer: "好" }],
    () => ids.shift(),
  );
  assert.equal(messages[1].id, "b1");
  assert.match(messages[1].error, /中断/);
  assert.equal(messages[2].error, undefined);
});

test("several conversations save side by side; one slow save does not hold up another", async () => {
  const log = [];
  let releaseA;
  const s = saver(async (c) => {
    log.push(`start ${c.id}`);
    if (c.id === "a") await new Promise((resolve) => { releaseA = resolve; });
    log.push(`end ${c.id}`);
  });
  s.save(conv("a"));
  s.save(conv("b"));
  await until(() => log.includes("end b") && releaseA);
  assert.ok(!log.includes("end a"));
  releaseA();
  await until(() => s.pending() === 0);
  assert.equal(log.at(-1), "end a");
});

test("saves of one conversation run in turn and the newest content wins", async () => {
  const seen = [];
  let running = 0;
  const c = conv("a", { n: 1 });
  const s = saver(async (item) => {
    running += 1;
    assert.equal(running, 1, "one at a time");
    seen.push(item.n);
    await tick(3);
    running -= 1;
  });
  s.save(c);
  await until(() => seen.length === 1);
  c.n = 2;
  s.save(c);
  c.n = 3;
  s.save(c);
  await until(() => s.pending() === 0);
  assert.equal(seen[0], 1);
  assert.equal(seen.at(-1), 3);
  assert.ok(seen.length <= 2, "saves while one runs collapse into one");
});

test("a failed save is retried until it works, and says so meanwhile", async () => {
  const states = [];
  let attempts = 0;
  const s = saver(
    async () => { attempts += 1; if (attempts < 3) throw new Error("backend away"); },
    async () => {},
    { onFailing: (failing) => states.push(failing) },
  );
  s.save(conv("a"));
  await until(() => s.pending() === 0);
  assert.equal(attempts, 3);
  assert.equal(states[0], true);
  assert.equal(states.at(-1), false);
});

test("a delete waits for the save in flight and later saves cannot bring it back", async () => {
  const log = [];
  let release;
  const s = saver(
    async (c) => { log.push("put"); await new Promise((resolve) => { release = resolve; }); log.push(`put done ${c.id}`); },
    async (id) => { log.push(`delete ${id}`); },
  );
  s.save(conv("a"));
  await until(() => release);
  s.remove("a");
  s.save(conv("a"));
  await tick();
  assert.deepEqual(log, ["put"]);
  release();
  await until(() => s.pending() === 0);
  assert.deepEqual(log, ["put", "put done a", "delete a"]);
});

test("flush sends what is waiting right away, marked as closing", async () => {
  const calls = [];
  const s = new Saver({ put: async (c, closing) => { calls.push([c.id, closing]); }, remove: async () => {}, delay: 60000 });
  s.save(conv("a"));
  await s.flush();
  assert.deepEqual(calls, [["a", true]]);
  assert.equal(s.pending(), 0);
});
