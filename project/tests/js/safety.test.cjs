// What 设置 › 关于 › 数据安全 says (project/ui/static/safety.js).
// test_ui_kit.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "safety.js"));
const { when, size, backupLine, summary } = globalThis.Safety;

const NOW = new Date(2026, 10, 5, 15, 0);

test("dates read as today, yesterday or the date", () => {
  assert.equal(when("2026-11-05T10:32:00", NOW), "今天 10:32");
  assert.equal(when("2026-11-04T22:05:00", NOW), "昨天 22:05");
  assert.equal(when("2026-11-03T09:07:00", NOW), "11月3日 09:07");
  assert.equal(when("", NOW), "");
  assert.equal(when("not a date", NOW), "");
});

test("sizes round to something readable", () => {
  assert.equal(size(0), "");
  assert.equal(size(2048), "2 KB");
  assert.equal(size(49 * 1024 * 1024), "49 MB");
  assert.equal(size(3 * 1024 ** 3), "3.0 GB");
});

test("a backup line says when, how big and why", () => {
  const backup = { created_at: "2026-11-05T10:32:00", size: 49 * 1024 * 1024, reason: "upgrade" };
  assert.equal(backupLine(backup, NOW), "今天 10:32 · 49 MB · 升级前");
  assert.equal(backupLine({ ...backup, reason: "auto" }, NOW), "今天 10:32 · 49 MB");
});

test("all well: the check passed and the newest backup is named", () => {
  const s = summary({
    state: "ok", checked_at: "2026-11-05T14:00:00",
    backups: [{ created_at: "2026-11-05T14:10:00" }, { created_at: "2026-11-04T14:10:00" }],
  }, NOW);
  assert.equal(s.damaged, false);
  assert.equal(s.check, "通过 · 今天 14:00");
  assert.equal(s.backups, "2 份 · 最新 今天 14:10");
  assert.equal(s.alert, "");
});

test("a damaged index says so plainly", () => {
  const s = summary({ state: "damaged", message: "索引库已损坏。原文件没有受影响。", backups: [] }, NOW);
  assert.equal(s.damaged, true);
  assert.equal(s.check, "没有通过");
  assert.match(s.alert, /原文件没有受影响/);
  assert.equal(s.backups, "还没有备份");
});

test("a backup that could not be made is what the row shows", () => {
  const s = summary({ state: "ok", backup_note: "磁盘空间不足，没有备份。", backups: [{ created_at: "2026-11-05T14:10:00" }] }, NOW);
  assert.equal(s.backups, "磁盘空间不足，没有备份。");
});

test("a recovery waiting for the next start says so", () => {
  assert.equal(summary({ state: "ok", pending: "restore", backups: [] }, NOW).pending, "重启后生效");
  assert.equal(summary({ state: "checking" }, NOW).check, "正在检查…");
  // The conversations went back to a backup: said plainly, with no recovery offered.
  const healed = summary({ state: "ok", notice: "对话记录文件损坏了", checked_at: "2026-11-05T09:00:00", backups: [] }, NOW);
  assert.equal(healed.damaged, false);
  assert.equal(healed.alert, "对话记录文件损坏了");
});
