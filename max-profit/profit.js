/**
 * Max Profit — core algorithm.
 *
 * Works both in the browser (exposes `window.MaxProfit`) and in Node
 * (`module.exports`) so the same code is used by the UI and the tests.
 */
(function (root) {
  "use strict";

  const BUILDINGS = [
    { code: "T", name: "Theatre", time: 5, rate: 1500 },
    { code: "P", name: "Pub", time: 4, rate: 1000 },
    { code: "C", name: "Commercial Park", time: 10, rate: 2000 },
  ];

  /**
   * Parse the time-unit input. Throws an Error with a readable message on invalid input.
   * @param {string} text
   * @returns {number}
   */
  function parseTime(text) {
    const cleaned = String(text).trim();
    if (cleaned === "") throw new Error("Please enter the number of time units.");
    if (!/^\d+$/.test(cleaned)) {
      throw new Error(`"${cleaned}" is not valid. Time units must be a whole number (0 or more).`);
    }
    return Number(cleaned);
  }

  /**
   * Find the maximum earnings after n time units, and every mix of buildings that achieves it.
   *
   * best[r] = max earnings when r time units remain:
   *   best[r] = max over buildings with time < r of  rate * (r - time) + best[r - time]
   *   best[r] = 0 when nothing can finish before time runs out.
   *
   * Time: O(n * k) = O(n) for k = 3 building types.   Space: O(n) for the table.
   *
   * @param {number} n
   * @returns {{ n: number, earnings: number, solutions: {T: number, P: number, C: number}[],
   *             best: number[], choices: number[][] }}
   */
  function maxProfit(n) {
    const best = new Array(n + 1).fill(0);
    const choices = Array.from({ length: n + 1 }, () => []);

    for (let r = 1; r <= n; r++) {
      BUILDINGS.forEach(({ time, rate }, k) => {
        if (time >= r) return; // finishing at or after n earns nothing
        const value = rate * (r - time) + best[r - time];
        if (value > best[r]) {
          best[r] = value;
          choices[r] = [k];
        } else if (value === best[r]) {
          choices[r].push(k);
        }
      });
    }

    // Trace back every tied choice, collecting distinct [T, P, C] counts.
    const memo = new Map();
    function mixes(r) {
      if (choices[r].length === 0) return [[0, 0, 0]];
      if (memo.has(r)) return memo.get(r);
      const seen = new Map();
      for (const k of choices[r]) {
        for (const mix of mixes(r - BUILDINGS[k].time)) {
          const next = mix.slice();
          next[k]++;
          seen.set(next.join(","), next);
        }
      }
      const result = [...seen.values()];
      memo.set(r, result);
      return result;
    }

    const solutions = mixes(n)
      .map(([T, P, C]) => ({ T, P, C }))
      .sort((a, b) => b.T - a.T || b.P - a.P || b.C - a.C);

    return { n, earnings: best[n], solutions, best, choices };
  }

  /**
   * Build order for a mix: highest earnings per unit of build time first (T 300 > P 250 > C 200),
   * which is the optimal order for any fixed mix.
   * @returns {{ code: string, name: string, start: number, finish: number, earnings: number }[]}
   */
  function schedule(solution, n) {
    const order = BUILDINGS.slice().sort((a, b) => b.rate / b.time - a.rate / a.time);
    const steps = [];
    let clock = 0;
    for (const b of order) {
      for (let i = 0; i < solution[b.code]; i++) {
        const start = clock;
        clock += b.time;
        steps.push({ code: b.code, name: b.name, start, finish: clock, earnings: b.rate * (n - clock) });
      }
    }
    return steps;
  }

  function formatSolution(s) {
    return `T: ${s.T} P: ${s.P} C: ${s.C}`;
  }

  const api = { BUILDINGS, parseTime, maxProfit, schedule, formatSolution };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.MaxProfit = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
