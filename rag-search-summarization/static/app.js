// Search UI: auto-suggest, paginated results, selectable results and adjustable-length summaries.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const form = $("search-form"), q = $("q"), list = $("suggestions"), method = $("method"), pageSize = $("page-size");
  const resultsPanel = $("results-panel"), results = $("results"), meta = $("results-meta"), pagination = $("pagination");
  const summaryPanel = $("summary-panel"), length = $("length"), lengthValue = $("length-value");
  const summarizeBtn = $("summarize"), summaryOut = $("summary-out"), selectedCount = $("selected-count");
  const errorEl = $("error"), reader = $("reader");

  let state = { query: "", page: 1, active: -1 };
  const selected = new Map(); // doc_id -> title, kept across pages

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  async function api(path, options) {
    const response = await fetch(path, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${response.status})`);
    return data;
  }

  function showError(message) { errorEl.textContent = message; errorEl.hidden = !message; }

  // ---------- auto-suggest ----------
  let suggestTimer;
  q.addEventListener("input", () => {
    clearTimeout(suggestTimer);
    suggestTimer = setTimeout(loadSuggestions, 150); // debounce: one request after typing pauses
  });

  async function loadSuggestions() {
    const text = q.value.trim();
    const lastWords = text.split(/\s+/).slice(-3).join(" "); // suggest for what is being typed now
    if (lastWords.length < 2) return hideSuggestions();
    try {
      const { suggestions } = await api(`/api/suggest?q=${encodeURIComponent(lastWords)}`);
      const prefix = text.slice(0, text.length - lastWords.length);
      renderSuggestions(suggestions.map((s) => prefix + s));
    } catch { hideSuggestions(); }
  }

  function renderSuggestions(items) {
    state.active = -1;
    list.innerHTML = items.map((s, i) => `<li role="option" id="sug-${i}" aria-selected="false">${esc(s)}</li>`).join("");
    list.hidden = items.length === 0;
    q.setAttribute("aria-expanded", String(!list.hidden));
  }

  function hideSuggestions() { list.hidden = true; q.setAttribute("aria-expanded", "false"); }

  list.addEventListener("mousedown", (e) => {
    if (e.target.tagName === "LI") { q.value = e.target.textContent; hideSuggestions(); runSearch(1); }
  });
  q.addEventListener("blur", () => setTimeout(hideSuggestions, 100));
  q.addEventListener("keydown", (e) => {
    const items = [...list.children];
    if (list.hidden || !items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      state.active = (state.active + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((li, i) => li.setAttribute("aria-selected", String(i === state.active)));
      q.setAttribute("aria-activedescendant", `sug-${state.active}`);
    } else if (e.key === "Enter" && state.active >= 0) {
      e.preventDefault(); q.value = items[state.active].textContent; hideSuggestions(); runSearch(1);
    } else if (e.key === "Escape") { hideSuggestions(); }
  });

  // ---------- search + pagination ----------
  form.addEventListener("submit", (e) => { e.preventDefault(); hideSuggestions(); runSearch(1); });
  method.addEventListener("change", () => state.query && runSearch(1));
  pageSize.addEventListener("change", () => state.query && runSearch(1));

  async function runSearch(page) {
    const query = q.value.trim();
    if (!query) return;
    const newQuery = query !== state.query;
    if (newQuery) selected.clear();
    state = { ...state, query, page };
    showError("");
    const url = `?q=${encodeURIComponent(query)}${page > 1 ? `&page=${page}` : ""}`;
    if (location.search !== url) history.replaceState(null, "", url); // shareable / bookmarkable searches
    try {
      const data = await api(`/api/search?q=${encodeURIComponent(query)}&method=${method.value}&page=${page}&page_size=${pageSize.value}`);
      if (newQuery) data.results.slice(0, 5).forEach((r) => selected.set(r.doc_id, r.title)); // pre-select top 5
      renderResults(data);
    } catch (err) { showError(err.message); }
  }

  // ~60-word window around the first query term, so each result stays short (full text: click the title).
  function snippet(text, query, size = 60) {
    const words = text.split(/\s+/);
    if (words.length <= size) return text;
    const terms = query.toLowerCase().split(/\W+/).filter((t) => t.length > 2);
    const hit = words.findIndex((w) => terms.some((t) => w.toLowerCase().replace(/\W/g, "").startsWith(t)));
    const start = Math.max(0, Math.min(hit < 0 ? 0 : hit - 15, words.length - size));
    return (start > 0 ? "… " : "") + words.slice(start, start + size).join(" ") + (start + size < words.length ? " …" : "");
  }

  function highlight(text, query) {
    const terms = query.toLowerCase().split(/\W+/).filter((t) => t.length > 2);
    let html = esc(text);
    if (!terms.length) return html;
    const pattern = new RegExp(`\\b(${terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w*`, "gi");
    return html.replace(pattern, "<mark>$&</mark>");
  }

  function renderResults(data) {
    resultsPanel.hidden = false;
    summaryPanel.hidden = data.total === 0;
    const t = data.timings_ms;
    meta.textContent = data.total
      ? `${data.total} documents · page ${data.page} of ${Math.ceil(data.total / data.page_size)} · ${t.total} ms`
      : "No matching documents.";
    results.innerHTML = data.results.map((r) => `
      <li class="result">
        <div class="result-head">
          <input type="checkbox" data-doc="${r.doc_id}" data-title="${esc(r.title)}" ${selected.has(r.doc_id) ? "checked" : ""}
                 aria-label="Include in summary">
          <h3>${r.rank}. <a data-open="${r.doc_id}">${esc(r.title)}</a></h3>
        </div>
        <p class="excerpt">${highlight(snippet(r.excerpt, data.query), data.query)}</p>
        <div class="meta">
          <span class="chip">score ${r.score}</span>
          ${r.matched_by.map((m) => `<span class="chip">${m === "bm25" ? "keyword match" : "semantic match"}</span>`).join("")}
          <span>${r.words} words</span>
        </div>
      </li>`).join("");
    renderPagination(data);
    updateSelectedCount();
  }

  function renderPagination(data) {
    const pages = Math.ceil(data.total / data.page_size);
    if (pages <= 1) { pagination.innerHTML = ""; return; }
    const show = [...new Set([1, data.page - 1, data.page, data.page + 1, pages])].filter((p) => p >= 1 && p <= pages);
    let html = `<button data-page="${data.page - 1}" ${data.page === 1 ? "disabled" : ""}>‹ Prev</button>`;
    show.forEach((p, i) => {
      if (i && p - show[i - 1] > 1) html += `<span class="muted">…</span>`;
      html += `<button data-page="${p}" ${p === data.page ? 'aria-current="page"' : ""}>${p}</button>`;
    });
    html += `<button data-page="${data.page + 1}" ${data.page === pages ? "disabled" : ""}>Next ›</button>`;
    pagination.innerHTML = html;
  }

  pagination.addEventListener("click", (e) => {
    const page = Number(e.target.dataset.page);
    if (page) { runSearch(page); window.scrollTo({ top: resultsPanel.offsetTop - 10, behavior: "smooth" }); }
  });

  results.addEventListener("change", (e) => {
    if (e.target.type !== "checkbox") return;
    const id = Number(e.target.dataset.doc);
    if (e.target.checked) selected.set(id, e.target.dataset.title); else selected.delete(id);
    updateSelectedCount();
  });
  results.addEventListener("click", (e) => { if (e.target.dataset.open) openDocument(Number(e.target.dataset.open)); });

  function updateSelectedCount() {
    selectedCount.textContent = selected.size;
    summarizeBtn.disabled = selected.size === 0 || selected.size > 10;
  }

  // ---------- summary ----------
  length.addEventListener("input", () => { lengthValue.textContent = length.value; });
  document.querySelectorAll(".presets button").forEach((b) => b.addEventListener("click", () => {
    length.value = b.dataset.words; lengthValue.textContent = b.dataset.words;
  }));

  summarizeBtn.addEventListener("click", async () => {
    summarizeBtn.disabled = true;
    summaryOut.innerHTML = '<p class="muted">Summarizing…</p>';
    try {
      const data = await api("/api/summarize", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: state.query, doc_ids: [...selected.keys()], length: Number(length.value) }),
      });
      const text = esc(data.summary).replace(/\[(\d+)\]/g, (m, n) =>
        `<a class="cite" data-open="${data.sources[n - 1] ?? ""}" title="${esc(data.source_titles[n - 1] || "")}">[${n}]</a>`);
      summaryOut.innerHTML = `
        <p class="summary-text">${text}</p>
        <p class="muted">${data.actual_words} words (target ${data.target_words}) · ${data.mode === "llm" ? esc(data.model) : "extractive fallback (LLM unavailable)"} · ${data.seconds}s</p>
        <ol class="sources">${data.source_titles.map((t, i) => `<li><a class="cite" data-open="${data.sources[i]}">${esc(t)}</a></li>`).join("")}</ol>`;
    } catch (err) { summaryOut.innerHTML = ""; showError(err.message); }
    finally { updateSelectedCount(); }
  });
  summaryOut.addEventListener("click", (e) => { if (e.target.dataset.open) openDocument(Number(e.target.dataset.open)); });

  // ---------- full article reader ----------
  async function openDocument(id) {
    try {
      const doc = await api(`/api/document/${id}`);
      $("reader-title").textContent = doc.title;
      $("reader-meta").textContent = `Document #${doc.doc_id} · ${doc.words} words · CNN/DailyMail ${doc.split} split`;
      $("reader-text").innerHTML = doc.text.match(/[^.!?]+[.!?]+["')\]]*\s*/g)
        .reduce((paras, s, i) => { if (i % 4 === 0) paras.push(""); paras[paras.length - 1] += s; return paras; }, [])
        .map((p) => `<p>${esc(p.trim())}</p>`).join("");
      reader.showModal();
    } catch (err) { showError(err.message); }
  }

  // Open a shared link like /?q=ebola&page=2
  const params = new URLSearchParams(location.search);
  if (params.get("q")) { q.value = params.get("q"); runSearch(Number(params.get("page")) || 1); }

  api("/api/health").then((h) => {
    $("corpus-info").textContent = `Hybrid keyword + semantic search over ${h.documents.toLocaleString()} CNN/DailyMail news articles (${h.chunks.toLocaleString()} passages), with AI summaries of the results.`;
  }).catch(() => {});
})();
