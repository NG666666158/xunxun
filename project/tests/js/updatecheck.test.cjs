// When the page looks for a new version by itself (project/ui/static/updatecheck.js).
// test_ui_kit.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "updatecheck.js"));
const { due, schedule, FIRST_MS, DAY_MS, RETRY_MS, TICK_MS } = globalThis.UpdateCheck;

const START = Date.UTC(2026, 10, 11, 9, 0, 0);
const MINUTE = 60 * 1000;
const HOUR = 60 * MINUTE;

test("once a little after the start, not before", () => {
  assert.equal(due({ startedAt: START }, START), false);
  assert.equal(due({ startedAt: START }, START + FIRST_MS - 1), false);
  assert.equal(due({ startedAt: START }, START + FIRST_MS), true);
});

test("after an answer, again a day later", () => {
  const s = { startedAt: START, lastTry: START + MINUTE, lastOk: START + MINUTE + 800 };
  assert.equal(due(s, START + MINUTE + 2 * HOUR), false);
  assert.equal(due(s, START + MINUTE + DAY_MS - 1), false);
  assert.equal(due(s, START + MINUTE + DAY_MS), true);
  // A PC that slept through the day checks when it wakes (wall clock, not timer time).
  assert.equal(due(s, START + MINUTE + 3 * DAY_MS), true);
});

test("after a failure, again an hour later", () => {
  const s = { startedAt: START, lastTry: START + DAY_MS, lastOk: START }; // last answer is older than the try
  assert.equal(due(s, START + DAY_MS + RETRY_MS - 1), false);
  assert.equal(due(s, START + DAY_MS + RETRY_MS), true);
  assert.equal(due({ startedAt: START, lastTry: START + MINUTE }, START + MINUTE + RETRY_MS), true); // never answered
});

test("never two at once, and a clock set back does not stop the checks", () => {
  assert.equal(due({ startedAt: START, busy: true }, START + DAY_MS), false);
  assert.equal(due({ startedAt: START, lastTry: START + DAY_MS, lastOk: START + DAY_MS }, START + HOUR), true);
});

/** A page with a hand-moved clock: ``every`` records the interval instead of starting it. */
function fakePage(check) {
  const clock = { t: START };
  const seen = [];
  const logs = [];
  let interval = null;
  const page = schedule({
    check,
    found: (update) => seen.push(update),
    log: (message) => logs.push(message),
    now: () => clock.t,
    every: (fn, ms) => { interval = { fn, ms }; },
  });
  return { clock, seen, logs, page, interval: () => interval };
}

test("the page asks every minute and checks once after the start, then daily", async () => {
  let calls = 0;
  const { clock, seen, page, interval } = fakePage(async () => (++calls === 1 ? null : { version: "1.0.3", notes: "" }));
  assert.equal(interval().ms, TICK_MS);
  assert.equal(await interval().fn(), false); // right at the start: too early
  clock.t += TICK_MS;
  assert.equal(await page.tick(), true);
  assert.deepEqual(seen, [null]); // up to date: the notice stays off
  clock.t += 12 * HOUR;
  assert.equal(await page.tick(), false);
  clock.t += 12 * HOUR;
  assert.equal(await page.tick(), true);
  assert.deepEqual(seen, [null, { version: "1.0.3", notes: "" }]);
  assert.equal(calls, 2);
});

test("a failure is only logged, and tried again an hour later", async () => {
  let fail = true;
  const { clock, seen, logs, page } = fakePage(async () => {
    if (fail) throw new Error("error sending request");
    return { version: "1.0.3" };
  });
  clock.t += TICK_MS;
  assert.equal(await page.tick(), true);
  assert.deepEqual(seen, []);
  assert.equal(logs.length, 1);
  assert.match(logs[0], /error sending request/);
  assert.equal(page.state.busy, false);
  clock.t += 30 * MINUTE;
  assert.equal(await page.tick(), false);
  fail = false;
  clock.t += 30 * MINUTE;
  assert.equal(await page.tick(), true);
  assert.deepEqual(seen, [{ version: "1.0.3" }]);
});

test("a slow check is not started a second time", async () => {
  let release;
  let calls = 0;
  const { clock, page } = fakePage(() => { calls++; return new Promise((resolve) => { release = resolve; }); });
  clock.t += TICK_MS;
  const first = page.tick();
  assert.equal(page.state.busy, true);
  clock.t += 2 * DAY_MS;
  assert.equal(await page.tick(), false);
  release(null);
  assert.equal(await first, true);
  assert.equal(calls, 1);
});
