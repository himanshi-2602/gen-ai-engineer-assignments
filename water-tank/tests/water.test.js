const test = require("node:test");
const assert = require("node:assert/strict");
const { computeWater, parseHeights } = require("../water.js");

test("example from the assignment stores 18 units", () => {
  const { total, water } = computeWater([0, 4, 0, 0, 0, 6, 0, 6, 4, 0]);
  assert.equal(total, 18);
  assert.deepEqual(water, [0, 0, 4, 4, 4, 0, 6, 0, 0, 0]);
});

test("classic LeetCode case", () => {
  assert.equal(computeWater([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]).total, 6);
  assert.equal(computeWater([4, 2, 0, 3, 2, 5]).total, 9);
});

test("no water when there are no walls on both sides", () => {
  assert.equal(computeWater([]).total, 0);
  assert.equal(computeWater([5]).total, 0);
  assert.equal(computeWater([3, 3]).total, 0);
  assert.equal(computeWater([1, 2, 3, 4]).total, 0);
  assert.equal(computeWater([4, 3, 2, 1]).total, 0);
  assert.equal(computeWater([0, 0, 0]).total, 0);
});

test("valley between equal walls", () => {
  assert.deepEqual(computeWater([3, 0, 0, 3]).water, [0, 3, 3, 0]);
});

test("matches brute force on random inputs", () => {
  const brute = (h) =>
    h.reduce((sum, x, i) => {
      const l = Math.max(...h.slice(0, i + 1));
      const r = Math.max(...h.slice(i));
      return sum + Math.min(l, r) - x;
    }, 0);
  for (let k = 0; k < 500; k++) {
    const h = Array.from({ length: 1 + Math.floor(Math.random() * 20) }, () =>
      Math.floor(Math.random() * 10)
    );
    assert.equal(computeWater(h).total, brute(h), `input ${JSON.stringify(h)}`);
  }
});

test("parseHeights accepts common formats", () => {
  assert.deepEqual(parseHeights("[0,4,0,0,0,6,0,6,4,0]"), [0, 4, 0, 0, 0, 6, 0, 6, 4, 0]);
  assert.deepEqual(parseHeights(" 1, 2  3 "), [1, 2, 3]);
});

test("parseHeights rejects invalid input", () => {
  assert.throws(() => parseHeights(""), /at least one/);
  assert.throws(() => parseHeights("1,-2,3"), /not valid/);
  assert.throws(() => parseHeights("1,a"), /not valid/);
  assert.throws(() => parseHeights("1.5"), /not valid/);
});
