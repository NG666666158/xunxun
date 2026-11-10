// Follow-up suggestions in the UI (project/ui/static/followups.js): keys, visibility, scope, cancelling.
// Run with `node --test project/tests/js`; test_followups_ui.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "followups.js"));
const { host, ghost, chips, keyAction, scopeChange, Requests } = globalThis.Followups;

const key = (k, extra = {}) => ({ key: k, isComposing: false, keyCode: 0, ...extra });
const items = [
  { text: "PagedAttention 是怎么解决 KV Cache 显存碎片的？", scope_action: "keep" },
  { text: "什么是投机解码？", scope_action: "keep" },
  { text: "《苏州三方协议》约定了哪些合作内容？", scope_action: "set", scope: { doc_ids: ["d1"], label: "苏州三方协议.docx" } },
  { text: "推理怎么加速到十倍？", scope_action: "clear", label: "全库" },
  { text: "第五个不显示？", scope_action: "keep" },
];
function conversation(extra = {}) {
  const older = { role: "bot", id: "b1", done: true, followups: { items, dismissed: false } };
  const last = { role: "bot", id: "b2", done: true, followups: { items, dismissed: false }, ...extra };
  return { id: "c1", messages: [{ role: "user" }, older, { role: "user" }, last] };
}

test("Tab and → fill the empty input, Enter sends, Esc dismisses", () => {
  const active = { value: "", active: true };
  assert.equal(keyAction(key("Tab"), active), "fill");
  assert.equal(keyAction(key("ArrowRight"), active), "fill");
  assert.equal(keyAction(key("Enter"), active), "send");
  assert.equal(keyAction(key("Escape"), active), "dismiss");
  assert.equal(keyAction(key("a"), active), null);
  // Shift+Enter is a new line, Shift+Tab moves focus back, Ctrl+→ is left alone.
  assert.equal(keyAction(key("Enter", { shiftKey: true }), active), null);
  assert.equal(keyAction(key("Tab", { shiftKey: true }), active), null);
  assert.equal(keyAction(key("ArrowRight", { ctrlKey: true }), active), null);
});

test("nothing happens once the user typed, or with no suggestion on offer", () => {
  assert.equal(keyAction(key("Enter"), { value: "我自己的问题", active: true }), null);
  assert.equal(keyAction(key("Tab"), { value: "x", active: true }), null);
  assert.equal(keyAction(key("Enter"), { value: "", active: false }), null);
  assert.equal(keyAction(key("Escape"), { value: "", active: false }), null);
});

test("keys of an input method that is composing are left to it", () => {
  const active = { value: "", active: true };
  assert.equal(keyAction(key("Enter", { isComposing: true }), active), null);
  assert.equal(keyAction(key("Escape", { isComposing: true }), active), null);
  assert.equal(keyAction(key("Tab", { isComposing: true }), active), null);
  // Chromium reports keyCode 229 for the key that starts a composition.
  assert.equal(keyAction(key("Process", { keyCode: 229 }), active), null);
  assert.equal(keyAction(key("Enter", { keyCode: 229 }), active), null);
});

test("only the conversation's last finished answer shows follow-ups", () => {
  const conv = conversation();
  assert.equal(host(conv).id, "b2");
  assert.equal(ghost(conv).text, items[0].text);
  // The alternatives below the answer: the next three, never the grey one again.
  assert.deepEqual(chips(conv).map((i) => i.text), items.slice(1, 4).map((i) => i.text));
  // A new question is running in this conversation: nothing.
  assert.equal(host(conv, { running: true }), null);
  assert.equal(ghost(conv, { running: true }), null);
  assert.deepEqual(chips(conv, { running: true }), []);
  // The last message is the user's (just sent) or a failed answer: nothing.
  assert.equal(host({ messages: [...conv.messages, { role: "user" }] }), null);
  assert.equal(host(conversation({ error: "请求失败" })), null);
  assert.equal(host(conversation({ done: false })), null);
  assert.equal(host(null), null);
});

test("the switch in Settings hides everything", () => {
  const conv = conversation();
  assert.equal(host(conv, { enabled: false }), null);
  assert.equal(ghost(conv, { enabled: false }), null);
  assert.deepEqual(chips(conv, { enabled: false }), []);
});

test("the grey suggestion goes once dismissed or while there is text, the chips stay", () => {
  assert.equal(ghost(conversation(), {}, "开"), null);
  const dismissed = conversation({ followups: { items, dismissed: true } });
  assert.equal(ghost(dismissed), null);
  assert.equal(chips(dismissed).length, 3);
});

test("a found file is asked about within that file; widening clears the scope", () => {
  assert.equal(scopeChange(items[0]), undefined);
  assert.deepEqual(scopeChange(items[2]), { doc_ids: ["d1"], label: "苏州三方协议.docx" });
  assert.equal(scopeChange(items[3]), null);
  assert.equal(scopeChange({ scope_action: "set", scope: { doc_ids: [] } }), undefined);
});

function deferredFetcher() {
  const calls = [];
  const fetcher = (body, signal) => new Promise((resolve, reject) => {
    calls.push({ body, signal, resolve });
    signal.addEventListener("abort", () => reject(Object.assign(new Error("aborted"), { name: "AbortError" })));
  });
  return { calls, fetcher };
}

test("typing or leaving the conversation cancels its request", async () => {
  const { calls, fetcher } = deferredFetcher();
  const requests = new Requests(fetcher);
  const pending = requests.start("c1", { thread_id: "c1" });
  assert.ok(requests.pending("c1"));
  assert.equal(requests.cancel("c1"), true);
  assert.equal(calls[0].signal.aborted, true);
  assert.equal(await pending, null);
  assert.equal(requests.pending("c1"), false);
  assert.equal(requests.cancel("c1"), false);
});

test("a newer request replaces the older one; other conversations are untouched", async () => {
  const { calls, fetcher } = deferredFetcher();
  const requests = new Requests(fetcher);
  const first = requests.start("c1", { n: 1 });
  const other = requests.start("c2", { n: 2 });
  const second = requests.start("c1", { n: 3 });
  assert.equal(calls[0].signal.aborted, true);
  assert.equal(calls[1].signal.aborted, false);
  calls[2].resolve({ items: ["新的"] });
  calls[1].resolve({ items: ["别的会话"] });
  assert.equal(await first, null);
  assert.deepEqual(await second, { items: ["新的"] });
  assert.deepEqual(await other, { items: ["别的会话"] });
  assert.equal(requests.pending("c1"), false);
});

test("a failed request shows nothing instead of an error", async () => {
  const requests = new Requests(async () => { throw new Error("无法连接后台服务"); });
  assert.equal(await requests.start("c1", {}), null);
  assert.equal(requests.pending("c1"), false);
});
