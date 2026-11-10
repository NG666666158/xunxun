// The interaction kit's pure parts (project/ui/static/ui.js, filerow.js): wheel snapping and
// momentum, tri-state selection, deck order, same-name file stacks.
// Run with `node --test project/tests/js`; test_ui_kit.py runs it under pytest.
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");

const STATIC = path.join(__dirname, "..", "..", "ui", "static");
require(path.join(STATIC, "ui.js"));
require(path.join(STATIC, "filerow.js"));
const { WheelMath, Selection, deckOrder } = globalThis.UI;
const { groupByName } = globalThis.FileRows;

test("the wheel snaps to the nearest row and never past either end", () => {
  assert.equal(WheelMath.nearest(0, 32, 5), 0);
  assert.equal(WheelMath.nearest(47, 32, 5), 1);
  assert.equal(WheelMath.nearest(49, 32, 5), 2);
  assert.equal(WheelMath.nearest(-200, 32, 5), 0);
  assert.equal(WheelMath.nearest(10_000, 32, 5), 4);
  assert.equal(WheelMath.nearest(10, 32, 0), 0);
});

test("a flick carries on in its direction, further the faster it was", () => {
  assert.equal(WheelMath.project(100, 0), 100);
  const slow = WheelMath.project(100, 0.5);
  const fast = WheelMath.project(100, 2);
  assert.ok(slow > 100 && fast > slow);
  assert.ok(WheelMath.project(100, -1) < 100);
  // A higher deceleration rate glides further.
  assert.ok(WheelMath.project(0, 1, 0.998) > WheelMath.project(0, 1, 0.99));
});

test("dragging past an end meets growing resistance", () => {
  assert.equal(WheelMath.rubber(50, 0, 128), 50);
  const a = WheelMath.rubber(-40, 0, 128);
  const b = WheelMath.rubber(-400, 0, 128);
  assert.ok(a < 0 && a > -40, "moves less than the finger");
  assert.ok(b < a && b > -120, "never further than the band");
  assert.ok(WheelMath.rubber(200, 0, 128) > 128 && WheelMath.rubber(200, 0, 128) < 200);
});

test("the spring settles on the target, with or without a starting velocity", () => {
  for (const v of [0, 1500, -2500]) {
    let state = { x: 0, v };
    let steps = 0;
    while (!WheelMath.settled(state, 96) && steps < 600) {
      state = WheelMath.step(state, 96, 1 / 120);
      steps++;
    }
    assert.ok(WheelMath.settled(state, 96), `settles from v=${v}`);
    assert.ok(steps < 300, `within 2.5 s (took ${steps} steps)`);
  }
});

test("mouse-wheel notches step whole rows; trackpads scroll freely", () => {
  assert.deepEqual(WheelMath.wheelDelta({ deltaMode: 0, deltaY: 100 }, 32), { px: 100, notch: true, dir: 1, steps: 1 });
  assert.equal(WheelMath.wheelDelta({ deltaMode: 0, deltaY: -300 }, 32).steps, 3);
  assert.equal(WheelMath.wheelDelta({ deltaMode: 0, deltaY: 125 }, 32).steps, 1);
  assert.equal(WheelMath.wheelDelta({ deltaMode: 1, deltaY: 3 }, 32).steps, 1);
  assert.equal(WheelMath.wheelDelta({ deltaMode: 0, deltaY: 4000 }, 32).steps, 5);
  const pad = WheelMath.wheelDelta({ deltaMode: 0, deltaY: 6.5 }, 32);
  assert.equal(pad.notch, false);
  assert.equal(pad.px, 6.5);
});

test("drum angles: the selected row faces front, neighbours turn away", () => {
  assert.equal(WheelMath.angle(2, 64, 32, 80), 0);
  assert.ok(WheelMath.angle(3, 64, 32, 80) > 0);
  assert.ok(WheelMath.angle(1, 64, 32, 80) < 0);
});

test("a folder is all, some or none selected, and follows its files", () => {
  const sel = new Selection([{ id: "a", items: ["a1", "a2", "a3"] }, { id: "b", items: ["b1"] }]);
  assert.equal(sel.groupState("a"), "none");
  sel.toggle("a1");
  assert.equal(sel.groupState("a"), "some");
  // A partly selected folder selects all on click, a full one clears.
  assert.equal(sel.toggleGroup("a"), true);
  assert.equal(sel.groupState("a"), "all");
  assert.equal(sel.size, 3);
  sel.toggleGroup("a");
  assert.equal(sel.groupState("a"), "none");
  assert.equal(sel.size, 0);

  sel.toggle("b1");
  sel.toggle("a2");
  assert.equal(sel.allState(), "some");
  assert.equal(sel.total, 4);
  assert.deepEqual(sel.ids().sort(), ["a2", "b1"]);
  sel.toggle("b1", true); // explicit on stays on
  assert.equal(sel.has("b1"), true);
  assert.equal(sel.groupState("missing"), "none");
});

test("a new file list keeps what is still there and drops what is gone", () => {
  const sel = new Selection([{ id: "a", items: ["a1", "a2"] }]);
  sel.toggleGroup("a");
  sel.setGroups([{ id: "x", items: ["a2", "c1"] }]);
  assert.deepEqual(sel.ids(), ["a2"]);
  assert.equal(sel.groupState("x"), "some");
  sel.clear();
  assert.equal(sel.allState(), "none");
});

test("a collapsed deck puts the peeked card on top", () => {
  assert.deepEqual(deckOrder(4), [0, 1, 2, 3]);
  assert.deepEqual(deckOrder(4, 2), [2, 0, 1, 3]);
  assert.deepEqual(deckOrder(4, 0), [0, 1, 2, 3]);
  assert.deepEqual(deckOrder(3, 9), [0, 1, 2]);
  assert.deepEqual(deckOrder(0), []);
});

test("files with the same name stack together in first-seen order", () => {
  const files = [
    { name: "01 申报书.pdf" }, { name: "报告.docx" }, { name: "01 申报书.PDF" }, { name: "清单.xlsx" }, { name: "报告.docx" },
  ];
  assert.deepEqual(groupByName(files), [[0, 2], [1, 4], [3]]);
  assert.deepEqual(groupByName([{ name: "" }, { name: "" }]), [[0], [1]]);
});
