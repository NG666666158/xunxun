// 只按文件名 / 不收录 as the page says it (project/ui/static/privacy.js). test_ui_kit.py runs it.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "privacy.js"));
const { options, groupMode, confirmText, undoAction, ruleRow, sortRules } = globalThis.PrivacyUI;

test("the choices: excluding only where a watched folder is around it", () => {
  assert.deepEqual(options().map((o) => o.value), ["full", "name"]);
  const folder = options({ kind: "folder", canExclude: true });
  assert.deepEqual(folder.map((o) => o.value), ["full", "name", "exclude"]);
  assert.match(folder[0].line, /发到硅基流动/);
  assert.match(folder[1].line, /不读取、不上传/);
  assert.match(folder[2].line, /以后新加的也不收/);
});

test("a folder is name-only, read, or mixed", () => {
  assert.equal(groupMode([]), "full");
  assert.equal(groupMode([{ content_mode: "name" }, { content_mode: "name" }]), "name");
  assert.equal(groupMode([{ content_mode: "name" }, { content_mode: "full" }]), "mixed");
  assert.equal(groupMode([{ content_mode: "full" }]), "full");
});

test("confirmations say plainly what is dropped and what is uploaded", () => {
  const name = confirmText("name", { kind: "folder", name: "证件", impact: { files: 23, indexed: 20 } });
  assert.equal(name.title, "「证件」只按文件名收录？");
  assert.equal(name.message, "里面 23 个文件和以后新加的，只在本机按名字、文件夹和日期找，内容不读取、不上传。\n其中 20 个已建的索引会删掉，备份里的文字也一并清除。");
  assert.equal(name.danger, false);
  // All of them indexed: no count to read.
  assert.equal(confirmText("name", { kind: "folder", name: "证件", impact: { files: 2, indexed: 2 } }).message.split("\n")[1],
    "已建的索引会删掉，备份里的文字也一并清除。");
  // Nothing indexed yet: no line about dropping.
  assert.equal(confirmText("name", { name: "护照.jpg", impact: { files: 1, indexed: 0 } }).message,
    "以后只在本机按名字、文件夹和日期找它，内容不读取、不上传。");

  const full = confirmText("full", { kind: "folder", name: "证件", impact: { files: 23, changed: 5 } });
  assert.equal(full.title, "读取「证件」的内容？");
  assert.equal(full.message, "5 个文件的文字会发到硅基流动建索引；回答问题时，相关段落也会发过去。");
  assert.equal(full.confirm, "读取并上传");
  assert.equal(confirmText("full", { name: "a.txt" }).confirm, "读取并上传");

  const exclude = confirmText("exclude", { kind: "pattern", name: "*.key", impact: { files: 3, indexed: 1 } });
  assert.equal(exclude.title, "不收录名字是 *.key 的文件？");
  assert.equal(exclude.message, "现有 3 个会从知识库移除，以后新加的也不收。\n其中 1 个已建的索引会删掉，备份里的文字也一并清除。\n原文件不动。");
  assert.equal(exclude.danger, true);

  // A pattern nothing matches yet: no "现有 0 个".
  assert.equal(confirmText("exclude", { kind: "pattern", name: "*.pem", impact: { files: 0 } }).message, "以后新加的这类文件不收录。\n原文件不动。");
  assert.equal(confirmText("name", { kind: "pattern", name: "证件", impact: { files: 0 } }).message,
    "以后新加的，只在本机按名字、文件夹和日期找，内容不读取、不上传。");
  const include = confirmText("include", { kind: "folder", name: "合同", impact: { files: 0 } });
  assert.equal(include.message, "现在没有这样的文件。以后新加的会照常收录。");
});

test("rules read as rows, exclusions first", () => {
  const rules = sortRules([
    { rule_id: "a", kind: "path", value: "证件", action: "name", root_id: "r", root_path: "D:\\资料", created_at: "1" },
    { rule_id: "b", kind: "pattern", value: "*.key", action: "exclude", root_id: null, root_path: null, created_at: "2" },
    { rule_id: "c", kind: "path", value: "工作\\合同", action: "exclude", root_id: "r", root_path: "D:\\资料", created_at: "3" },
  ]);
  assert.deepEqual(rules.map((r) => r.rule_id), ["b", "c", "a"]);
  assert.deepEqual(ruleRow(rules[0]), { icon: "eyeOff", label: "*.key", line: "不收录 · 所有文件夹" });
  assert.equal(ruleRow({ kind: "pattern", value: "*.pfx", action: "name", root_id: "r", root_path: "D:\\资料" }).line, "只按文件名 · 只在「资料」");
  assert.deepEqual(ruleRow(rules[1]), { icon: "eyeOff", label: "合同", line: "不收录 · 资料 › 工作" });
  assert.deepEqual(ruleRow(rules[2]), { icon: "tag", label: "证件", line: "只按文件名 · 资料" });
  // The watched folder itself kept by name only.
  assert.equal(ruleRow({ kind: "path", value: "", action: "name", root_id: "r", root_path: "D:\\资料" }).label, "资料");
  assert.equal(undoAction({ action: "exclude" }), "include");
  assert.equal(undoAction({ action: "name" }), "full");
  assert.equal(undoAction({ action: "full" }), "name");
});
