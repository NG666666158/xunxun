// The status row, ahead-of-time preview preparation and 寻寻's fetching lines
// (project/ui/static/status.js). test_ui_kit.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "status.js"));
const { describe, jobLines, PrepareQueue, fetchLine, homeMode, cardTag } = globalThis.AppStatus;

test("files read side by side: one line each, a scan with its pages", () => {
  const jobs = [
    { name: "01 申报书.pdf", pages: [16, 61], kind: "long" },
    { name: "白板.png", pages: null, kind: "ocr" },
    { name: "", pages: null },
  ];
  assert.deepEqual(jobLines(jobs), [
    { name: "01 申报书.pdf", detail: "16 / 61 页", progress: 16 / 61 },
    { name: "白板.png", detail: "", progress: null },
  ]);
  const s = describe({ connected: true, keyConfigured: true, ready: true,
    library: { busy: true, batch: { total: 20, finished: 5 }, jobs } });
  assert.equal(s.text, "正在读 15 个文件");
  assert.equal(s.title, "01 申报书.pdf（16 / 61 页）\n白板.png");
});

const healthy = { connected: true, keyConfigured: true, ready: true };

test("all well is a quiet green dot", () => {
  assert.deepEqual(describe(healthy), { tone: "ok", text: "", target: "" });
  assert.equal(describe({ ...healthy, library: { busy: false } }).tone, "ok");
});

test("problems come first, each pointing where it can be fixed", () => {
  const busy = { busy: true, batch: { total: 10, finished: 2 } };
  assert.deepEqual(
    [
      describe({ ...healthy, connected: false, keyConfigured: false, library: busy }),
      describe({ ...healthy, keyConfigured: false, ready: false, library: busy }),
      describe({ ...healthy, ready: false, readyError: "向量库打不开", modelIssue: "模型连接失败" }),
      describe({ ...healthy, modelIssue: "模型连接失败", library: busy }),
    ].map((s) => [s.tone, s.text, s.target]),
    [
      ["error", "后台没有响应", "about"],
      ["error", "还没有 Key", "key"],
      ["error", "服务未就绪", "about"],
      ["error", "模型连接失败", "key"],
    ],
  );
  assert.equal(describe({ ...healthy, ready: false, readyError: "向量库打不开" }).title, "向量库打不开");
});

test("while files are being read it says how many are left, with progress", () => {
  const s = describe({ ...healthy, library: { busy: true, batch: { total: 12, finished: 3 }, current: { name: "报告.pdf" } } });
  assert.equal(s.tone, "busy");
  assert.equal(s.text, "正在读 9 个文件");
  assert.equal(s.progress, 0.25);
  assert.equal(s.target, "library");
  assert.equal(s.title, "报告.pdf");
  const scanning = describe({ ...healthy, library: { busy: true, batch: { total: 0, finished: 0 } } });
  assert.equal(scanning.text, "正在清点文件夹");
  assert.equal(scanning.progress, null);
});

test("before the first check it is still connecting", () => {
  assert.equal(describe({}).tone, "wait");
  assert.equal(describe({ connected: true, keyConfigured: true }).text, "正在连接…");
});

test("only Word files are prepared, each once, in small batches", async () => {
  const sent = [];
  const queue = new PrepareQueue((ids) => { sent.push(ids); }, { delay: 10_000, batch: 2 });
  assert.equal(queue.add([
    { id: "a", name: "合同.docx" }, { id: "b", name: "报告.pdf" }, { id: "c", name: "旧版.DOC" },
    { id: "a", name: "合同.docx" }, { id: "", name: "x.docx" }, { id: "d", name: "清单.xlsx" }, { id: "e", name: "说明.docx" },
  ]), 3);
  queue.flush();
  assert.deepEqual(sent, [["a", "c"], ["e"]]);
  assert.equal(queue.add([{ id: "a", name: "合同.docx" }]), 0);
  queue.flush();
  assert.equal(sent.length, 2);
});

test("a batch the backend did not take may be asked for again", async () => {
  let fail = true;
  const sent = [];
  const queue = new PrepareQueue(async (ids) => { sent.push(ids); if (fail) throw new Error("busy"); }, { delay: 10_000 });
  queue.add([{ id: "a", name: "a.docx" }]);
  queue.flush();
  await new Promise((r) => setTimeout(r, 0));
  fail = false;
  assert.equal(queue.add([{ id: "a", name: "a.docx" }]), 1);
  queue.flush();
  assert.deepEqual(sent, [["a"], ["a"]]);
});

test("the queue sends on its own after a short delay", async () => {
  const sent = [];
  const queue = new PrepareQueue((ids) => { sent.push(ids); }, { delay: 5 });
  queue.add([{ id: "a", name: "a.docx" }]);
  queue.add([{ id: "b", name: "b.docx" }]);
  assert.deepEqual(sent, []);
  await new Promise((r) => setTimeout(r, 30));
  assert.deepEqual(sent, [["a", "b"]]);
});

test("寻寻's line moves on the longer a file takes", () => {
  assert.equal(fetchLine(0), "寻寻找到文件踪迹");
  assert.equal(fetchLine(1500), "正在把文件叼回来");
  assert.equal(fetchLine(5000), "快到了，再等一下");
  assert.equal(fetchLine(20_000), "这个文件有点大，寻寻还在叼");
  for (const ms of [0, 1500, 5000, 20_000]) assert.doesNotMatch(fetchLine(ms), /Word|PDF|转换/);
});

test("the home shows the input in its middle only once the library has something to ask", () => {
  assert.equal(homeMode({ conversation: true }), "off");
  assert.equal(homeMode({ conversation: true, libraryEmpty: true }), "off");
  assert.equal(homeMode({ libraryEmpty: true }), "start");
  assert.equal(homeMode({ libraryEmpty: false }), "focal");
  assert.equal(homeMode(), "focal");
});

test("a card names why it is there only when that says something", () => {
  assert.equal(cardTag({ kind: "new", label: "新加入" }), "新加入");
  assert.equal(cardTag({ kind: "continue", label: "接着上次的话题" }), "接着上次的话题");
  assert.equal(cardTag({ kind: "gist", label: "摘要" }), "");
  assert.equal(cardTag({ kind: "new" }), "");
  assert.equal(cardTag(null), "");
});
