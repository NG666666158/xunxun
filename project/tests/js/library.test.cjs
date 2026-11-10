// Adding files and folders, the image choice, and the status row's notes about changes
// (project/ui/static/library.js and status.js). test_ui_kit.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "library.js"));
require(path.join(__dirname, "..", "..", "ui", "static", "status.js"));
const { duration, imageOptions, plan, importSummary, changeNote, lostLine, isImage } = globalThis.LibraryUI;
const { describe } = globalThis.AppStatus;

test("durations read as people say them", () => {
  assert.equal(duration(12), "不到 1 分钟");
  assert.equal(duration(7 * 60 + 10), "约 7 分钟");
  assert.equal(duration(5400), "约 1.5 小时");
  assert.equal(duration(14 * 3600), "约 14 小时");
  assert.equal(duration(null), "");
});

test("the image choice says plainly that cloud uploads, and what local needs first", () => {
  const estimate = { local: { seconds: 75 }, cloud: { seconds: 400, cloud_calls: 31 }, name: { seconds: 5 } };
  const [local, cloud, name] = imageOptions(estimate, { ocrInstalled: false, ocrMB: 77 });
  assert.equal(local.value, "local");
  assert.equal(local.line, "识别图中文字，不上传 · 约 1 分钟 · 先下载约 77 MB");
  assert.ok(cloud.line.startsWith("图片会上传到硅基流动") && cloud.warn);
  assert.equal(name.value, "name");
  // No key: cloud cannot be picked; no estimate: no dangling separators.
  const [plain, noKey] = imageOptions({}, { keyConfigured: false });
  assert.equal(plain.line, "识别图中文字，不上传");
  assert.ok(noKey.disabled);
});

test("a batch plan: what to add, what to ask, when to confirm", () => {
  const inspect = {
    items: [
      { path: "D:\\资料", kind: "folder", ok: true, already: false },
      { path: "C:\\a.png", kind: "file", ok: true, already: false },
      { path: "C:\\old.md", kind: "file", ok: true, already: true },
      { path: "C:\\x.exe", kind: "file", ok: false, error: "不支持的格式：.exe" },
    ],
    totals: { indexable: 40, images: 3 },
    estimate: { text_seconds: 60, images: { local: { seconds: 30 }, cloud: { seconds: 200 }, name: { seconds: 2 } } },
    image_mode: "ask",
    ask_image_mode: true,
    default_image_mode: "local",
  };
  const p = plan(inspect);
  assert.deepEqual(p.add.map((i) => i.path), ["D:\\资料", "C:\\a.png"]);
  assert.equal(p.already.length, 1);
  assert.equal(p.errors[0].error, "不支持的格式：.exe");
  assert.equal(p.askImages, true);
  assert.equal(p.seconds, 90); // text + the default (local) image time
  assert.equal(p.large, false);
  // Remembered choice: no question; a huge folder is confirmed.
  const big = plan({ ...inspect, image_mode: "cloud", ask_image_mode: false, totals: { indexable: 5000, images: 3 } });
  assert.equal(big.askImages, false);
  assert.equal(big.seconds, 260);
  assert.equal(big.large, true);
  // Nothing new: no question either.
  assert.equal(plan({ ...inspect, items: [inspect.items[2]] }).askImages, false);
});

test("after adding: one line with what went in and the first reason something did not", () => {
  const result = {
    results: [
      { kind: "folder", ok: true, already: false },
      { kind: "file", ok: true, already: false },
      { kind: "file", ok: true, already: false },
      { kind: "folder", ok: true, already: true },
      { kind: "error", ok: false, error: "文件或文件夹不存在：D:\\x" },
    ],
  };
  const s = importSummary(result, [{ ok: false, error: "不支持的格式：.exe" }]);
  assert.equal(s.text, "已添加 1 个文件夹和 2 个文件，1 个已经在知识库里，2 个没加上：不支持的格式：.exe");
  assert.equal(s.ok, true);
  assert.equal(importSummary({ results: [] }).text, "没有要添加的");
});

test("change notes: losses first and plain; news in 寻寻's voice", () => {
  assert.deepEqual(changeNote({ new: 3, missing: 1 }), { tone: "warn", text: "1 个文件找不到了", target: "lost" });
  assert.deepEqual(changeNote({ recycled: 2 }), { tone: "warn", text: "2 个文件在回收站", target: "lost" });
  assert.equal(changeNote({ recycled: 1, missing: 1 }).text, "2 个文件找不到了");
  assert.equal(changeNote({ folder_recycled: 1 }).text, "有个文件夹在回收站");
  assert.deepEqual(changeNote({ new: 3 }, true), { tone: "busy", text: "寻寻发现 3 个新文件，正在读", target: "library" });
  assert.equal(changeNote({ new: 3 }).tone, "note");
  assert.equal(changeNote({ moved: 2 }).text, "寻寻跟上了 2 个换位置的文件");
  assert.equal(changeNote({ folder_moved: 1 }).text, "寻寻跟上了换位置的文件夹");
  assert.equal(changeNote({ restored: 1 }).text, "1 个文件回来了");
  assert.equal(changeNote({}), null);
});

test("the status row shows a note under problems, with reading progress, or on its own", () => {
  const healthy = { connected: true, keyConfigured: true, ready: true };
  const busy = { busy: true, batch: { total: 10, finished: 4 } };
  const news = { tone: "busy", text: "寻寻发现 3 个新文件，正在读", target: "library" };
  assert.equal(describe({ ...healthy, keyConfigured: false, note: news }).text, "还没有 Key");
  const reading = describe({ ...healthy, library: busy, note: news });
  assert.deepEqual([reading.tone, reading.text, reading.progress], ["busy", "寻寻发现 3 个新文件，正在读", 0.4]);
  const lost = describe({ ...healthy, library: busy, note: { tone: "warn", text: "1 个文件找不到了", target: "lost" } });
  assert.deepEqual([lost.tone, lost.target], ["warn", "lost"]);
  assert.deepEqual(describe({ ...healthy, note: { tone: "note", text: "1 个文件回来了" } }), { tone: "note", text: "1 个文件回来了", target: "library" });
  assert.equal(describe({ ...healthy, library: { ...busy, paused: true } }).title, "先回答你的问题，答完接着读");
});

test("lost files say where they were", () => {
  assert.equal(lostLine({ status: "recycled", path: "C:\\桌面\\a.md" }), "在回收站 · 还原后会自动回来");
  assert.equal(lostLine({ status: "missing", path: "D:\\资料\\合同\\a.pdf" }), "原来在 D:\\资料\\合同");
  assert.equal(lostLine({ status: "offline", path: "E:\\备份\\b.pdf" }), "磁盘未连接 · E:\\备份");
  assert.ok(isImage("白板.JPG") && isImage("scan.tiff") && !isImage("a.pdf"));
});

test("a list that does not show everything says so and offers the next page", () => {
  const { moreFiles, FILE_PAGE, FILE_MAX } = globalThis.LibraryUI;
  assert.equal(moreFiles(100, 100), null);
  assert.equal(moreFiles(0, 0), null);
  const more = moreFiles(1100, 500);
  assert.equal(more.step, FILE_PAGE);
  assert.match(more.text, /已显示 500 个，共 1,100 个文件/);
  const capped = moreFiles(FILE_MAX + 1, FILE_MAX);
  assert.equal(capped.step, 0);
  assert.match(capped.text, /搜索或筛选/);
});

test("a long list is not redrawn on every change while files are being read", () => {
  const { refreshDue } = globalThis.LibraryUI;
  assert.equal(refreshDue({ stale: false, busy: false, rows: 10, sinceLoad: 99999 }), false);
  assert.equal(refreshDue({ stale: true, busy: true, rows: 50, sinceLoad: 0 }), true);
  assert.equal(refreshDue({ stale: true, busy: true, rows: 500, sinceLoad: 1200 }), false);
  assert.equal(refreshDue({ stale: true, busy: true, rows: 500, sinceLoad: 15000 }), true);
  // The last change is always shown once reading has finished.
  assert.equal(refreshDue({ stale: true, busy: false, rows: 500, sinceLoad: 0 }), true);
});
