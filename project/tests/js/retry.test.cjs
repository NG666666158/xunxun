"use strict";
// node --test project/tests/js/retry.test.cjs

const test = require("node:test");
const assert = require("node:assert/strict");

require("../../ui/static/retry.js");
const { plan, failed, countdown, AUTO_SECONDS } = globalThis.Retry;

test("an overloaded or unreachable service is asked again once by itself", () => {
  for (const kind of ["busy", "rate_limit", "network", "server"]) {
    assert.deepEqual(plan(kind, { retryable: true }), { action: "retry", auto: AUTO_SECONDS });
    assert.deepEqual(plan(kind, { retryable: true, autoUsed: true }), { action: "retry", auto: 0 });
  }
});

test("an unknown failure offers asking again, but not by itself", () => {
  assert.deepEqual(plan("unknown", { retryable: true }), { action: "retry", auto: 0 });
  assert.deepEqual(plan("", {}), { action: "retry", auto: 0 }); // older backends say nothing
});

test("a bad key or no balance sends to the settings", () => {
  assert.deepEqual(plan("auth", { retryable: false }), { action: "settings", auto: 0 });
  assert.deepEqual(plan("balance", {}), { action: "settings", auto: 0 });
  assert.deepEqual(plan("bad_request", { retryable: false }), { action: null, auto: 0 });
});

test("which answers count as failed", () => {
  assert.equal(failed({ role: "bot", error: "问答服务暂时不可用，请稍后重试。" }), true);
  assert.equal(failed({ role: "bot", error: "已停止生成。" }), false);
  assert.equal(failed({ role: "bot", done: true, refused: true, errorKind: "busy" }), true);
  assert.equal(failed({ role: "bot", done: true, refused: true, errorKind: "" }), false); // no evidence is not a failure
  assert.equal(failed({ role: "user", text: "问题" }), false);
});

test("the countdown line", () => {
  assert.equal(countdown(9.2), "10 秒后自动再问一次");
  assert.equal(countdown(-1), "0 秒后自动再问一次");
});
