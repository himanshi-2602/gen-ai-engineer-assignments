/**
 * Water Tank (Trapping Rain Water) — core algorithm.
 *
 * Works both in the browser (exposes `window.WaterTank`) and in Node
 * (`module.exports`) so the same code is used by the UI and the tests.
 */
(function (root) {
  "use strict";

  /**
   * Parse user input like "0,4,0,0,6" or "[0, 4, 0]" into an array of heights.
   * Throws an Error with a readable message on invalid input.
   * @param {string} text
   * @returns {number[]}
   */
  function parseHeights(text) {
    const cleaned = String(text).trim().replace(/^\[|\]$/g, "").trim();
    if (cleaned === "") throw new Error("Please enter at least one block height.");

    return cleaned.split(/[\s,]+/).filter(Boolean).map((token) => {
      if (!/^\d+$/.test(token)) {
        throw new Error(`"${token}" is not valid. Heights must be whole numbers greater than -1.`);
      }
      return Number(token);
    });
  }

  /**
   * Compute water trapped between blocks using the two-pointer technique.
   *
   * Water above column i = min(maxLeft(i), maxRight(i)) - height[i].
   * We walk inward from both ends; the side with the smaller running max is
   * the limiting wall, so that column's water level is already known.
   *
   * Time: O(n)   Space: O(n) for the per-column result (O(1) extra otherwise).
   *
   * @param {number[]} heights
   * @returns {{ total: number, water: number[] }}
   */
  function computeWater(heights) {
    const n = heights.length;
    const water = new Array(n).fill(0);
    let left = 0;
    let right = n - 1;
    let leftMax = 0;
    let rightMax = 0;
    let total = 0;

    while (left < right) {
      if (heights[left] < heights[right]) {
        leftMax = Math.max(leftMax, heights[left]);
        water[left] = leftMax - heights[left];
        total += water[left];
        left++;
      } else {
        rightMax = Math.max(rightMax, heights[right]);
        water[right] = rightMax - heights[right];
        total += water[right];
        right--;
      }
    }

    return { total, water };
  }

  const api = { parseHeights, computeWater };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.WaterTank = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
