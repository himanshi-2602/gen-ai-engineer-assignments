/**
 * UI layer: reads input, runs the algorithm, and renders SVG grids.
 */
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const CELL = 40; // px size of one unit cell (scaled down by CSS on small screens)
  const MAX_COLUMNS = 200;
  const MAX_HEIGHT = 100;

  const form = document.getElementById("input-form");
  const input = document.getElementById("heights");
  const errorEl = document.getElementById("error");
  const results = document.getElementById("results");
  const totalEl = document.getElementById("total");
  const unitsEl = document.getElementById("units");
  const chartFull = document.getElementById("chart-full");
  const chartWater = document.getElementById("chart-water");

  function el(name, attrs) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    return node;
  }

  /**
   * Draw a grid where each column stacks `block` cells then `water` cells.
   * @param {number[]} heights
   * @param {number[]} water
   * @param {boolean} showBlocks  false renders the "water only" view
   */
  function renderGrid(heights, water, showBlocks) {
    const cols = heights.length;
    // One empty row on top (like the assignment diagram), at least one row overall.
    const rows = Math.max(1, ...heights.map((h, i) => h + water[i])) + 1;
    const width = cols * CELL;
    const height = rows * CELL;

    const svg = el("svg", {
      viewBox: `0 0 ${width} ${height + 24}`,
      width,
      role: "img",
      "aria-label": showBlocks ? "Blocks with trapped water" : "Trapped water only",
    });

    for (let i = 0; i < cols; i++) {
      const blockH = showBlocks ? heights[i] : 0;
      const x = i * CELL;

      if (blockH > 0) {
        svg.appendChild(el("rect", {
          x, y: height - blockH * CELL, width: CELL, height: blockH * CELL, class: "cell-block",
        }));
      }
      if (water[i] > 0) {
        const top = height - (heights[i] + water[i]) * CELL;
        svg.appendChild(el("rect", {
          x, y: top, width: CELL, height: water[i] * CELL, class: "cell-water",
        }));
      }

      const label = el("text", { x: x + CELL / 2, y: height + 17, class: "col-label" });
      label.textContent = showBlocks ? heights[i] : water[i];
      svg.appendChild(label);
    }

    // Grid lines drawn last so they sit on top of the filled cells.
    for (let r = 0; r <= rows; r++) {
      svg.appendChild(el("line", { x1: 0, y1: r * CELL, x2: width, y2: r * CELL, class: "grid" }));
    }
    for (let c = 0; c <= cols; c++) {
      svg.appendChild(el("line", { x1: c * CELL, y1: 0, x2: c * CELL, y2: height, class: "grid" }));
    }

    return svg;
  }

  function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
    results.hidden = true;
  }

  function run() {
    let heights;
    try {
      heights = WaterTank.parseHeights(input.value);
    } catch (err) {
      showError(err.message);
      return;
    }
    if (heights.length > MAX_COLUMNS) {
      showError(`Please enter at most ${MAX_COLUMNS} blocks.`);
      return;
    }
    if (Math.max(...heights) > MAX_HEIGHT) {
      showError(`Block heights must be at most ${MAX_HEIGHT} for the visualization.`);
      return;
    }

    errorEl.hidden = true;
    const { total, water } = WaterTank.computeWater(heights);

    totalEl.textContent = total;
    unitsEl.textContent = total === 1 ? "Unit" : "Units";
    chartFull.replaceChildren(renderGrid(heights, water, true));
    chartWater.replaceChildren(renderGrid(heights, water, false));
    results.hidden = false;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    run();
  });

  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = `[${button.dataset.example}]`;
      run();
    });
  });

  run();
})();
