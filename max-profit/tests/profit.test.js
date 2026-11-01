const test = require("node:test");
const assert = require("node:assert/strict");
const { BUILDINGS, parseTime, maxProfit, schedule, formatSolution } = require("../profit.js");

const formatted = (n) => maxProfit(n).solutions.map(formatSolution);

test("test case 1: n = 7 earns $3000 with two solutions", () => {
  assert.equal(maxProfit(7).earnings, 3000);
  assert.deepEqual(formatted(7), ["T: 1 P: 0 C: 0", "T: 0 P: 1 C: 0"]);
});

test("test case 2: n = 8 earns $4500", () => {
  assert.equal(maxProfit(8).earnings, 4500);
  assert.deepEqual(formatted(8), ["T: 1 P: 0 C: 0"]);
});

test("test case 3: n = 13 earns $16500", () => {
  assert.equal(maxProfit(13).earnings, 16500);
  assert.deepEqual(formatted(13), ["T: 2 P: 0 C: 0"]);
});

test("ties are all reported (n = 9)", () => {
  assert.equal(maxProfit(9).earnings, 6000);
  assert.deepEqual(formatted(9), ["T: 1 P: 0 C: 0", "T: 0 P: 2 C: 0"]);
});

test("nothing can finish in time for n <= 4", () => {
  for (const n of [0, 1, 4]) {
    assert.equal(maxProfit(n).earnings, 0);
    assert.deepEqual(formatted(n), ["T: 0 P: 0 C: 0"]);
  }
  assert.equal(maxProfit(5).earnings, 1000); // one Pub finishing at 4 earns for 1 unit
});

test("schedule of every solution adds up to the max earnings", () => {
  for (let n = 0; n <= 120; n++) {
    const { earnings, solutions } = maxProfit(n);
    for (const s of solutions) {
      const steps = schedule(s, n);
      assert.equal(steps.reduce((sum, st) => sum + st.earnings, 0), earnings, `n=${n} ${formatSolution(s)}`);
      assert.ok(steps.every((st) => st.finish < n), "every building finishes before n");
    }
  }
});

// Independent check: try every build sequence (exponential, so small n only).
function bruteForce(n) {
  let bestEarnings = 0;
  let mixes = new Set(["0,0,0"]);
  (function explore(clock, earned, counts) {
    if (earned > bestEarnings) {
      bestEarnings = earned;
      mixes = new Set();
    }
    if (earned === bestEarnings) mixes.add(counts.join(","));
    BUILDINGS.forEach(({ time, rate }, k) => {
      const finish = clock + time;
      if (finish >= n) return;
      const next = counts.slice();
      next[k]++;
      explore(finish, earned + rate * (n - finish), next);
    });
  })(0, 0, [0, 0, 0]);
  return { earnings: bestEarnings, mixes: [...mixes].sort() };
}

test("matches brute force over every build order for n = 0..40", () => {
  for (let n = 0; n <= 40; n++) {
    const dp = maxProfit(n);
    const brute = bruteForce(n);
    assert.equal(dp.earnings, brute.earnings, `earnings for n=${n}`);
    assert.deepEqual(dp.solutions.map((s) => `${s.T},${s.P},${s.C}`).sort(), brute.mixes, `mixes for n=${n}`);
  }
});

test("handles large n quickly", () => {
  const { earnings, solutions } = maxProfit(10000);
  assert.ok(earnings > 0);
  assert.ok(solutions.length >= 1);
});

test("parseTime validates input", () => {
  assert.equal(parseTime(" 13 "), 13);
  assert.equal(parseTime("0"), 0);
  assert.throws(() => parseTime(""), /enter/);
  assert.throws(() => parseTime("-3"), /not valid/);
  assert.throws(() => parseTime("2.5"), /not valid/);
  assert.throws(() => parseTime("abc"), /not valid/);
});
