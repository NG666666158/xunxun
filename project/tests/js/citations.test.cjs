// Inline citations in the UI (project/ui/static/citations.js): marker forms, matching, drafts.
// Run with `node --test project/tests/js`; test_citations_ui.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

require(path.join(__dirname, "..", "..", "ui", "static", "citations.js"));
const { link, strip } = globalThis.Citations;

const citations = [
  { source: "苏州三方协议.docx", locator: "未标注位置", doc_id: "d1", quote: "甲方…" },
  { source: "差旅制度.docx", locator: "section:2", doc_id: "d2" },
  { source: "差旅制度.docx", locator: "section:5", doc_id: "d3" },
  { source: "vLLM 推理加速.pdf", locator: "page:3", doc_id: "d4" },
];
const cited = (result) => result.sources.filter((s) => s.cited).map((s) => [s.n, s.doc_id]);
const chips = (result) => result.text.replace(/\u0000(\d+)\u0000/g, (_, k) => `{${result.marks[Number(k)]}}`);

test("a locator without its 位置 label links to its card", () => {
  for (const marker of [
    "[来源：苏州三方协议.docx | 未标注位置]",
    "[来源：苏州三方协议.docx｜未标注位置]",
    "【来源：苏州三方协议.docx，未标注位置】",
    "[来源: 苏州三方协议.docx ; 未标注位置]",
  ]) {
    const result = link(`三方约定按季度结算${marker}。`, citations);
    assert.equal(chips(result), "三方约定按季度结算{1}。", marker);
    assert.deepEqual(cited(result), [[1, "d1"]], marker);
    assert.equal(result.sources.length, citations.length, marker);
  }
});

test("labelled markers still link by source and locator, with every separator", () => {
  for (const sep of ["｜", "|", "，", ",", ";", "；"]) {
    const result = link(`住宿标准见制度[来源：差旅制度.docx${sep}位置：section:5]`, citations);
    assert.deepEqual(cited(result), [[1, "d3"]], sep);
  }
  const result = link("加速十倍[来源：vLLM 推理加速.pdf｜位置：page:3]，见下[来源：苏州三方协议.docx｜位置：未标注位置]", citations);
  assert.equal(chips(result), "加速十倍{1}，见下{2}");
  assert.deepEqual(cited(result), [[1, "d4"], [2, "d1"]]);
  assert.deepEqual(result.sources.map((s) => s.n), [1, 2, 3, 4]);
  assert.equal(result.sources[2].cited, false);
});

test("spacing and case are ignored; an unknown locator falls back to the source's first card", () => {
  assert.deepEqual(cited(link("[来源：VLLM推理加速.pdf｜位置：Page:3]", citations)), [[1, "d4"]]);
  assert.deepEqual(cited(link("[来源：差旅制度.docx｜位置：第9节]", citations)), [[1, "d2"]]);
  assert.deepEqual(cited(link("[来源：差旅制度.docx]", citations)), [[1, "d2"]]);
  // A file name the model shortened.
  assert.deepEqual(cited(link("[来源：苏州三方协议｜位置：未标注位置]", citations)), [[1, "d1"]]);
});

test("the same card cited twice keeps one number", () => {
  const result = link("甲[来源：差旅制度.docx，section:2] 乙[来源：差旅制度.docx｜位置：section:2]", citations);
  assert.equal(chips(result), "甲{1} 乙{1}");
  assert.equal(result.sources.filter((s) => s.cited).length, 1);
});

test("old answers citing a removed source and passages that were not retrieved get cards of their own", () => {
  const old = link("质保期为 3 年 [来源：结构化数据库]。", [{ source: "结构化数据库", locator: "" }]);
  assert.equal(chips(old), "质保期为 3 年 {1}。");
  assert.deepEqual(old.sources.map((s) => [s.n, s.source, s.cited]), [[1, "结构化数据库", true]]);

  const result = link("甲[来源：其他.pdf｜位置：page:1] 乙[来源：其他.pdf，page:1] 丙[来源：其他.pdf｜位置：page:2]", citations);
  assert.equal(chips(result), "甲{1} 乙{1} 丙{2}");
  assert.deepEqual(result.sources.slice(0, 2).map((s) => [s.source, s.locator, s.doc_id, s.cited]), [
    ["其他.pdf", "page:1", "", true],
    ["其他.pdf", "page:2", "", true],
  ]);
  assert.equal(result.sources.length, citations.length + 2);
});

test("a source name with a separator in it is not split when it names a card", () => {
  const cards = [{ source: "报告,2024.pdf", locator: "page:1", doc_id: "r" }];
  assert.deepEqual(cited(link("[来源：报告,2024.pdf]", cards)), [[1, "r"]]);
});

test("an answer without markers cites nothing", () => {
  const result = link("知识库中没有找到相关依据。", citations);
  assert.equal(result.text, "知识库中没有找到相关依据。");
  assert.deepEqual(result.marks, []);
  assert.equal(result.sources.some((s) => s.cited), false);
  assert.deepEqual(link("", null), { text: "", marks: [], sources: [] });
});

test("drafts hide complete markers in every form and one still being typed", () => {
  assert.equal(strip("按季度结算[来源：苏州三方协议.docx | 未标注位置]。"), "按季度结算。");
  assert.equal(strip("见制度[来源：差旅制度.docx｜位置：section:5]，另见"), "见制度，另见");
  assert.equal(strip("质保期为 3 年 [来源：结构化数据库]。"), "质保期为 3 年 。");
  assert.equal(strip("按季度结算[来源：杭州三方"), "按季度结算");
  assert.equal(strip("按季度结算【来源"), "按季度结算");
  assert.equal(strip("没有引用的草稿"), "没有引用的草稿");
});

test("slide and sheet locators read as places in the file", () => {
  require(path.join(__dirname, "..", "..", "ui", "static", "filerow.js"));
  const { locatorLabel } = globalThis.FileRows;
  assert.equal(locatorLabel("slide:3"), "第 3 张");
  assert.equal(locatorLabel("slide:3-5"), "第 3–5 张");
  assert.equal(locatorLabel("page:7"), "第 7 页");
  assert.equal(locatorLabel("sheet:科技研发;rows:2-13"), "科技研发 · 第 2–13 行");
});
