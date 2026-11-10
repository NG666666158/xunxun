// What the guided start shows and how a key test reads (project/ui/static/guide.js).
// test_ui_kit.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "guide.js"));
const { steps, keyOutcome, looksLikeKey, MODELS, NOTICE, KEY_URL } = globalThis.Guide;

test("a new install gets every step, starting with where data goes", () => {
  assert.deepEqual(steps({}), ["notice", "key", "model", "folder"]);
});

test("someone upgrading with a key and files only sees where data goes, once", () => {
  assert.deepEqual(steps({ keyConfigured: true, libraryEmpty: false }), ["notice"]);
  assert.deepEqual(steps({ keyConfigured: true, libraryEmpty: false, noticeSeen: true }), []);
});

test("a finished or skipped guide never comes back, except for the notice", () => {
  assert.deepEqual(steps({ onboardingDone: true, noticeSeen: true }), []);
  assert.deepEqual(steps({ onboardingDone: true }), ["notice"]);
});

test("a key from before, with nothing added yet, goes straight to the first folder", () => {
  assert.deepEqual(steps({ keyConfigured: true, noticeSeen: true, libraryEmpty: true }), ["folder"]);
});

test("a working key is saved; a bad one is not", () => {
  assert.deepEqual(keyOutcome(""), { ok: true, save: true, tone: "ok", text: "可以用，已保存到 Windows 凭据管理器。" });
  const bad = keyOutcome("Key 无效或已被停用。");
  assert.equal(bad.ok, false);
  assert.equal(bad.save, false);
  assert.equal(bad.tone, "error");
});

test("a valid key on an empty account is kept, with a plain warning", () => {
  const empty = keyOutcome("Key 有效，但硅基流动账户余额不足，请先充值。");
  assert.equal(empty.save, true);
  assert.equal(empty.ok, true);
  assert.equal(empty.tone, "warn");
  assert.match(empty.text, /余额不足/);
});

test("obviously incomplete keys are stopped before a request", () => {
  assert.equal(looksLikeKey("sk-abcdef12"), true);
  assert.equal(looksLikeKey("sk-abc"), false);
  assert.equal(looksLikeKey("sk-abc def12"), false);
});

test("the recommended model comes first and the free one says so", () => {
  assert.equal(MODELS[0].value, "Qwen/Qwen3-30B-A3B-Instruct-2507");
  assert.match(MODELS[0].line, /^推荐/);
  assert.match(MODELS.find((m) => m.value === "Qwen/Qwen3-8B").line, /^免费/);
});

test("the notice names what goes where, in plain words", () => {
  const text = NOTICE.map((f) => `${f.title} ${f.line}`).join(" ");
  assert.match(text, /硅基流动/);
  assert.match(text, /文档文字/);
  assert.match(text, /文件名/);
  assert.match(text, /问题/);
  assert.match(text, /没有本地模型/);
  assert.match(text, /只按文件名|不收录/); // and what to do about files that must not go
  assert.doesNotMatch(text, /寻寻/); // privacy notes stay out of the pet's voice
  assert.match(KEY_URL, /^https:\/\/cloud\.siliconflow\.cn\//);
});
