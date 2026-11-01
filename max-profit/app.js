/**
 * UI layer: reads n, runs the DP, and renders solutions, an SVG build timeline and the DP table.
 */
(function () {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const MAX_TIME = 10000;
  const MAX_TIMELINE_ROWS = 40;
  const MAX_TABLE_ROWS = 200;
  const ROW_H = 26;
  const LABEL_W = 150;
  const CHART_W = 640;
  const AXIS_H = 26;

  const form = document.getElementById("input-form");
  const input = document.getElementById("time");
  const errorEl = document.getElementById("error");
  const results = document.getElementById("results");
  const earningsEl = document.getElementById("earnings");
  const solutionsEl = document.getElementById("solutions");
  const timelinesEl = document.getElementById("timelines");
  const tableEl = document.getElementById("dp-table");

  const money = (value) => `$${value.toLocaleString("en-US")}`;

  function el(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /** One row per building: a hatched construction bar, then a solid bar while it earns. */
  function renderTimeline(steps, n) {
    const scale = CHART_W / Math.max(n, 1);
    const height = steps.length * ROW_H + AXIS_H;
    const svg = el("svg", {
      viewBox: `0 0 ${LABEL_W + CHART_W + 90} ${height}`,
      role: "img",
      "aria-label": "Build timeline",
    });

    steps.forEach((step, i) => {
      const y = i * ROW_H + 4;
      const h = ROW_H - 8;
      svg.appendChild(el("text", { x: 0, y: y + h - 4, class: "row-label" }, `${i + 1}. ${step.name}`));
      svg.appendChild(el("rect", {
        x: LABEL_W + step.start * scale, y, width: (step.finish - step.start) * scale, height: h, class: "bar-build",
      }));
      svg.appendChild(el("rect", {
        x: LABEL_W + step.finish * scale, y, width: (n - step.finish) * scale, height: h, class: `bar-${step.code}`,
      }));
      svg.appendChild(el("text", { x: LABEL_W + CHART_W + 8, y: y + h - 4, class: "row-value" }, money(step.earnings)));
    });

    // Time axis with a handful of ticks.
    const axisY = steps.length * ROW_H + 4;
    svg.appendChild(el("line", { x1: LABEL_W, y1: axisY, x2: LABEL_W + CHART_W, y2: axisY, class: "axis" }));
    const tickStep = Math.max(1, Math.ceil(n / 13));
    for (let t = 0; t <= n; t += tickStep) {
      const x = LABEL_W + t * scale;
      svg.appendChild(el("line", { x1: x, y1: axisY, x2: x, y2: axisY + 4, class: "axis" }));
      svg.appendChild(el("text", { x, y: axisY + 18, class: "tick" }, t));
    }
    return svg;
  }

  function renderTimelines(solutions, n) {
    timelinesEl.replaceChildren();
    solutions.forEach((solution, i) => {
      const steps = MaxProfit.schedule(solution, n);
      const block = document.createElement("div");
      block.className = "timeline";

      const title = document.createElement("h3");
      title.textContent = `Solution ${i + 1}: ${MaxProfit.formatSolution(solution)}`;
      block.appendChild(title);

      if (steps.length === 0) {
        block.appendChild(Object.assign(document.createElement("p"), {
          className: "note",
          textContent: "No building can be finished before time runs out, so nothing earns money.",
        }));
      } else if (steps.length > MAX_TIMELINE_ROWS) {
        block.appendChild(Object.assign(document.createElement("p"), {
          className: "note",
          textContent: `${steps.length} buildings, too many to draw. Try a smaller n to see the timeline.`,
        }));
      } else {
        block.appendChild(renderTimeline(steps, n));
      }
      timelinesEl.appendChild(block);
    });
  }

  function renderTable({ n, best, choices }) {
    if (n + 1 > MAX_TABLE_ROWS) {
      tableEl.replaceChildren(Object.assign(document.createElement("p"), {
        className: "note",
        textContent: `The table has ${n + 1} rows; it is shown for n < ${MAX_TABLE_ROWS} only.`,
      }));
      return;
    }
    const table = document.createElement("table");
    table.innerHTML = "<thead><tr><th>r (time left)</th><th>best[r]</th><th>Best first build</th></tr></thead>";
    const body = document.createElement("tbody");
    for (let r = 0; r <= n; r++) {
      const row = document.createElement("tr");
      const choice = choices[r].map((k) => MaxProfit.BUILDINGS[k].code).join(" or ") || "—";
      for (const text of [r, money(best[r]), choice]) {
        const cell = document.createElement("td");
        cell.textContent = text;
        row.appendChild(cell);
      }
      if (r === n) row.className = "highlight";
      body.appendChild(row);
    }
    table.appendChild(body);
    tableEl.replaceChildren(table);
  }

  function showError(message) {
    errorEl.textContent = message;
    errorEl.hidden = false;
    results.hidden = true;
  }

  function run() {
    let n;
    try {
      n = MaxProfit.parseTime(input.value);
    } catch (err) {
      showError(err.message);
      return;
    }
    if (n > MAX_TIME) {
      showError(`Please enter at most ${MAX_TIME} time units.`);
      return;
    }

    errorEl.hidden = true;
    const result = MaxProfit.maxProfit(n);

    earningsEl.textContent = money(result.earnings);
    solutionsEl.replaceChildren(...result.solutions.map((s) => {
      const li = document.createElement("li");
      li.textContent = MaxProfit.formatSolution(s);
      return li;
    }));
    renderTimelines(result.solutions, n);
    renderTable(result);
    results.hidden = false;
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    run();
  });

  document.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = button.dataset.example;
      run();
    });
  });

  run();
})();
