/* Offline deck: data.js is generated from published v28 results. */
(() => {
  "use strict";
  const data = window.SLIDES_DATA;
  const colors = {selected: "#2a78d6", reference: "#eb6834", baseline: "#1baf7a", other: "#c9c8c1"};
  const chosen = "sticklim_e5_d0+stock";
  const runnerUp = "lim_e5_d1+stock";
  const f3a = "ref_lim_f3a";
  const selection = Object.fromEntries(data.selection.map(row => [row.recipe, row]));
  const confirmation = Object.fromEntries(data.confirmation.map(row => [row.recipe, row]));
  const labels = {
    [chosen]: "Stick+Lim + E, +600 it",
    [runnerUp]: "Lim + E + D, +600 it",
    [f3a]: "Lim F3a (팀원 원본)",
    "lim_e5_d0+stock": "Lim + E, +600 it",
    "lim_e5_d1+recovery": "Lim + E + D, +600 it 회복",
    "sticklim_e5_d0": "Stick+Lim + E",
    "lim_e5_d1": "Lim + E + D",
    "stick_e5_d0": "Stick + E",
    "sticklim_e5_d1": "Stick+Lim + E + D",
    "flat_e0_d1": "이전 제출 (Robust42)",
    "flat_e0_d0": "제공 모델",
  };
  const shortLabel = recipe => ({[chosen]: "선택 조합", [runnerUp]: "2위 조합", [f3a]: "Lim F3a"}[recipe] || labels[recipe]);
  const color = recipe => recipe === chosen ? colors.selected : recipe.startsWith("ref_") ? colors.reference : recipe.startsWith("flat_e0_d") ? colors.baseline : colors.other;
  const escape = value => String(value).replace(/[&<>"']/g, char => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
  // Decimal half-up display for CSV values such as 43.35; no double rounding.
  const fixed = (value, digits = 1) => Number(value).toLocaleString("en-US", {minimumFractionDigits: digits, maximumFractionDigits: digits, useGrouping: false});
  const signed = (value, digits = 1) => `${value >= 0 ? "+" : "−"}${fixed(Math.abs(value), digits)}`;
  const text = (x, y, value, attrs = "") => `<text x="${x}" y="${y}" ${attrs}>${escape(value)}</text>`;
  const line = (x1, y1, x2, y2, attrs = 'class="grid"') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${attrs}/>`;
  const rect = (x, y, width, height, fill, attrs = "") => `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${fill}" ${attrs}/>`;
  function chart(id, width, height, title, content) {
    document.getElementById(id).innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-labelledby="${id}-title"><title id="${id}-title">${escape(title)}</title>${content}</svg>`;
  }

  // Visible numeric claims and every chart are bound to generated source data.
  for (const element of document.querySelectorAll("[data-value]")) {
    const keys = element.dataset.value.split(".");
    const source = {selection, confirmation, baseline_problem: data.baseline_problem, protocol: data.protocol, official: data.official};
    const value = keys.reduce((current, key) => current[key], source);
    element.textContent = fixed(value, Number(element.dataset.digits || 0));
  }
  const effectText = {
    "randomization-flat": signed(data.effects.randomization.category_means.flat_friction),
    "continuation-range": `${signed(Math.min(...data.effects.continuation.map(row => row.delta)))} ~ ${signed(Math.max(...data.effects.continuation.map(row => row.delta)))}`,
    "recovery-range": `${signed(Math.max(...data.effects.recovery.map(row => row.delta)))} ~ ${signed(Math.min(...data.effects.recovery.map(row => row.delta)))}`,
  };
  const recovery = data.effects.recovery.find(row => row.stock === runnerUp);
  effectText["fall-change"] = `${fixed(recovery.stock_fall_rate * 100, 0)}%에서 ${fixed(recovery.recovery_fall_rate * 100, 0)}%`;
  for (const element of document.querySelectorAll("[data-effect]")) element.textContent = effectText[element.dataset.effect];
  document.getElementById("category-cards").innerHTML = data.categories.map(category => `<div class="category-card"><p>${escape(category.label.replace("·", ", "))}</p><span class="category-count">${category.count}<small>조건</small></span></div>`).join("");
  document.querySelector('[data-ladder="entropy-delta"]').textContent = signed(data.ladder[2].delta);
  const countWord = number => ({1: "한", 2: "두", 3: "세"}[number] || String(number));
  const ladderSteps = data.ladder.slice(1);
  document.querySelector('[data-ladder="wins-summary"]').textContent = ladderSteps.every(row => row.wins === row.pairs && row.pairs === ladderSteps[0].pairs)
    ? `각 단계에서 학습 ${countWord(ladderSteps[0].pairs)} 번 모두 점수가 올랐습니다.`
    : ladderSteps.map((row, index) => `${index + 1}단계는 ${row.pairs}번 중 ${row.wins}번 올랐습니다.`).join(" ");
  const alternativeLabels = [
    ["이전 모델의 랜덤화", "entropy 조합과 비교했습니다."],
    ["회복 보상으로 추가 학습", "원래 보상의 추가 학습과 비교했습니다."],
    ["장애물, 계단, 급경사 지형", "이전 험지 세트로 Stick+Lim을 바꿨습니다."],
  ];
  document.getElementById("alternatives-rows").innerHTML = data.alternatives.map((row, index) => `<tr><td>${alternativeLabels[index][0]}<small>${alternativeLabels[index][1]}</small></td><td>${fixed(row.mean)}</td><td>${signed(row.delta)}</td><td>${row.wins === 0 ? "모두 낮아졌습니다." : `${row.pairs}번 중 ${row.wins}번 올랐습니다.`}</td></tr>`).join("");
  const alternativeRecovery = data.alternatives[1];
  document.querySelector('[data-alternative="fall-change"]').textContent = `${fixed(alternativeRecovery.reference_fall_rate * 100, 0)}%에서 ${fixed(alternativeRecovery.fall_rate * 100, 0)}%`;

  function renderLadder() {
    const labels = ["제공 모델", "험지 학습", "entropy 0.005", "600 iteration 추가"];
    const centers = [265, 665, 1065, 1465], barWidth = 215, bottom = 378;
    const y = value => bottom - value / 70 * 315;
    let svg = line(105, bottom, 1670, bottom);
    for (const tick of [0, 20, 40, 60]) {
      svg += text(70, y(tick) + 10, tick, 'class="axis-label" text-anchor="end"');
      svg += line(105, y(tick), 1670, y(tick));
    }
    data.ladder.forEach((row, index) => {
      const left = centers[index] - barWidth / 2;
      svg += rect(left, y(row.mean), barWidth, bottom - y(row.mean), index === 0 ? colors.baseline : index === 3 ? colors.selected : colors.other);
      svg += text(centers[index], y(row.mean) - 18, fixed(row.mean), 'class="ladder-number" text-anchor="middle"');
      svg += text(centers[index], 428, labels[index], 'text-anchor="middle"');
      if (index === 1) svg += text(centers[index], 471, "높이 관측도 바닥 기준", 'class="axis-label" text-anchor="middle"');
      if (index > 0) {
        const previous = data.ladder[index - 1];
        svg += line(centers[index - 1] + barWidth / 2 + 4, y(previous.mean), left - 4, y(previous.mean), 'stroke="#767c72" stroke-dasharray="6 5" stroke-width="2"');
        svg += text((centers[index - 1] + centers[index]) / 2, y(previous.mean) + 57, signed(row.delta), 'class="key-number" text-anchor="middle"');
      }
    });
    chart("ladder-chart", 1728, 505, `개선 과정: ${data.ladder.map(row => fixed(row.mean)).join(", ")}.`, svg);
  }

  function renderRanking() {
    const rows = [...data.ranking].sort((a, b) => b.demo_mean - a.demo_mean);
    const left = 505, plot = 1030, top = 25, rowHeight = 46;
    const x = value => left + value / 75 * plot;
    let svg = "";
    for (const tick of [0, 20, 40, 60]) {
      svg += line(x(tick), 5, x(tick), 535);
      svg += text(x(tick), 566, tick, 'class="axis-label" text-anchor="middle"');
    }
    rows.forEach((row, index) => {
      const y = top + index * rowHeight;
      svg += text(0, y + 25, labels[row.recipe] || row.label);
      svg += rect(left, y, x(row.demo_mean) - left, 33, color(row.recipe));
      svg += text(x(row.demo_mean) + 17, y + 30, fixed(row.demo_mean), 'class="ranking-number"');
    });
    svg += text(1710, 566, "데모 점수", 'class="axis-label" text-anchor="end"');
    chart("ranking-chart", 1728, 595, "새 조합 상위 8개와 기준 모델의 평균 점수. F3a는 모델 하나를 평가했습니다.", svg);
  }

  function renderEffects() {
    const effects = [["terrain", "평지 대신 팀원 지형"], ["entropy", "entropy 적용"], ["randomization", "랜덤화 적용"]];
    const x = value => 350 + value / 20 * 470;
    let svg = "";
    for (const tick of [0, 5, 10, 15, 20]) {
      svg += line(x(tick), 0, x(tick), 270);
      svg += text(x(tick), 319, tick, 'class="axis-label" text-anchor="middle"');
    }
    effects.forEach(([key, label], index) => {
      const row = data.effects[key], y = index * 94 + 18;
      svg += text(0, y + 25, label);
      svg += text(0, y + 63, `${row.pairs}개 중 ${row.wins}개가 올랐습니다.`, 'class="axis-label"');
      svg += rect(350, y, x(row.mean) - 350, 42, colors.other);
      svg += text(x(row.mean) + 14, y + 45, signed(row.mean), 'class="key-number"');
    });
    chart("effects-chart", 1019, 350, "다른 조건을 고정했을 때 학습 지형, entropy, 랜덤화의 점수 차이", svg);
  }

  function renderValidation(id, rows) {
    const left = 170, plot = 515, x = value => left + value / 70 * plot;
    let svg = "";
    for (const tick of [0, 20, 40, 60]) {
      svg += line(x(tick), 0, x(tick), 205);
      svg += text(x(tick), 237, tick, 'class="axis-label" text-anchor="middle"');
    }
    rows.forEach((row, index) => {
      const y = 27 + index * 65;
      svg += text(0, y + 30, shortLabel(row.recipe));
      svg += rect(left, y, x(row.value) - left, 39, color(row.recipe));
      svg += text(x(row.value) + 15, y + 33, fixed(row.value), 'class="key-number"');
    });
    svg += text(815, 237, "점수", 'class="axis-label" text-anchor="end"');
    if (id === "confirmation-chart") {
      svg += text(0, 285, `Robust42 ${fixed(confirmation.flat_e0_d1.demo_mean)}, 제공 모델 ${fixed(confirmation.flat_e0_d0.demo_mean)}`, 'class="axis-label"');
    }
    chart(id, 826, 292, `${id === "fresh-chart" ? "모델마다 따로 실행" : "새 지형 평가"} 28조건 평균: ${rows.map(row => `${shortLabel(row.recipe)} ${fixed(row.value)}`).join(", ")}`, svg);
  }

  function tint(hex, fraction) {
    const components = hex.match(/[a-f\d]{2}/gi).map(channel => Math.round(255 * (1 - fraction) + parseInt(channel, 16) * fraction));
    return `rgb(${components.join(",")})`;
  }
  function renderHeatmap() {
    const rows = [chosen, f3a, "flat_e0_d1", "flat_e0_d0"].map(key => selection[key]);
    const left = 348, width = 220, top = 90, height = 73;
    let svg = "";
    data.categories.forEach((category, index) => {
      svg += text(left + index * width + width / 2, 44, category.label.replace("·", ", "), 'text-anchor="middle"');
      svg += text(left + index * width + width / 2, 79, `${category.count}조건`, 'class="axis-label" text-anchor="middle"');
    });
    rows.forEach((row, y) => {
      const label = shortLabel(row.recipe);
      svg += text(0, top + y * height + 43, label);
      data.categories.forEach((category, x) => {
        const value = row.category_means[category.key];
        svg += rect(left + x * width, top + y * height, width - 6, height - 6, tint("#737870", value / 150 * .35));
        svg += text(left + x * width + (width - 6) / 2, top + y * height + 43, fixed(value), 'text-anchor="middle"');
      });
    });
    svg += text(350, 444, "0부터 150점까지 같은 명암을 썼습니다.", 'class="axis-label"');
    chart("heatmap-chart", 1728, 475, "범주별 평균 return. 선택한 조합, Lim F3a, Robust42, 제공 모델을 비교했습니다.", svg);
  }

  function renderLearning() {
    const points = data.learning_curves.points;
    const x = value => 100 + value / 1600 * 1500;
    const y = value => 347 - (value + 5) / 85 * 290;
    let svg = "";
    for (const tick of [0, 20, 40, 60, 80]) {
      svg += line(100, y(tick), 1600, y(tick));
      svg += text(75, y(tick) + 10, tick, 'class="axis-label" text-anchor="end"');
    }
    for (const tick of [0, 400, 800, 1000, 1200, 1600]) svg += text(x(tick), 405, tick, 'class="axis-label" text-anchor="middle"');
    svg += text(100, 31, "학습 reward", 'class="axis-label"');
    svg += text(1720, 405, "it", 'class="axis-label" text-anchor="end"');
    svg += line(x(1000), 48, x(1000), 350, 'stroke="#767c72" stroke-dasharray="8 8" stroke-width="2"');
    svg += text(x(1000) + 17, 34, "원래 보상으로 600 it 추가 학습", 'class="axis-label"');
    // Keep the optimizer restart boundary disconnected instead of interpolating it.
    for (const stage of [points.filter(p => p.iteration < 1000), points.filter(p => p.iteration >= 1000)]) {
      const band = [...stage.map(p => `${x(p.iteration)},${y(p.max)}`), ...[...stage].reverse().map(p => `${x(p.iteration)},${y(p.min)}`)].join(" ");
      svg += `<polygon points="${band}" fill="#dedfd9"/>`;
      svg += `<polyline points="${stage.map(p => `${x(p.iteration)},${y(p.mean)}`).join(" ")}" fill="none" stroke="#4b514b" stroke-width="4"/>`;
      const endpoint = stage[stage.length - 1];
      svg += `<circle cx="${x(endpoint.iteration)}" cy="${y(endpoint.mean)}" r="6" fill="#4b514b"/>`;
      svg += endpoint.iteration < 1000
        ? text(x(endpoint.iteration) - 15, y(endpoint.mean) - 25, fixed(endpoint.mean), 'text-anchor="end"')
        : text(x(endpoint.iteration) + 15, y(endpoint.mean) + 8, fixed(endpoint.mean));
    }
    const restart = points.find(point => point.iteration === 1000);
    svg += line(x(restart.iteration) + 8, y(restart.mean), 1090, 284, 'stroke="#767c72" stroke-width="2"');
    svg += text(1110, 265, "새 run 시작 직후", 'class="axis-label"');
    svg += text(1110, 306, "통계가 초기화됐습니다.", 'class="axis-label"');
    svg += text(1110, 347, "성능 하락은 아니었습니다.", 'class="axis-label"');
    chart("learning-chart", 1728, 445, "선택한 조합을 세 번 학습한 평균 reward와 최솟값, 최댓값. 1000 iteration에서 새 run의 통계가 초기화됐습니다. 평가 점수 하락은 아니었습니다.", svg);
  }

  renderRanking();
  renderLadder();
  renderEffects();
  renderValidation("confirmation-chart", [chosen, runnerUp, f3a].map(recipe => ({recipe, value: confirmation[recipe].demo_mean})));
  renderValidation("fresh-chart", data.fresh.map(row => ({recipe: row.recipe, value: row.fresh_demo})));
  renderHeatmap();
  renderLearning();

  const slides = [...document.querySelectorAll(".slide")];
  const previous = document.getElementById("previous");
  const next = document.getElementById("next");
  const notesToggle = document.getElementById("notes-toggle");
  const notesPanel = document.getElementById("notes-panel");
  let current = 0;
  let notesOpen = false;
  const duration = seconds => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  function fit() {
    const height = Math.max(100, window.innerHeight - (notesOpen ? 240 : 0));
    document.getElementById("deck").style.setProperty("--scale", Math.min(window.innerWidth / 1920, height / 1080));
  }
  function show(index) {
    current = Math.max(0, Math.min(slides.length - 1, index));
    slides.forEach((slide, i) => {
      const active = i === current;
      slide.classList.toggle("is-active", active);
      slide.setAttribute("aria-hidden", String(!active));
      if (!active) for (const video of slide.querySelectorAll("video")) video.pause();
    });
    // aria-hidden is screen-only; print CSS exposes every slide visually.
    document.getElementById("position").textContent = `${current + 1} / ${slides.length}`;
    const progress = document.getElementById("progress");
    progress.setAttribute("aria-valuenow", current + 1);
    progress.setAttribute("aria-valuemax", slides.length);
    progress.firstElementChild.style.width = `${(current + 1) / slides.length * 100}%`;
    previous.disabled = current === 0;
    next.disabled = current === slides.length - 1;
    const mainSeconds = slides.filter(slide => !slide.classList.contains("appendix")).reduce((sum, slide) => sum + Number(slide.dataset.noteSeconds), 0);
    document.getElementById("notes-heading").textContent = `${current + 1} / ${slides.length}, 예상 ${slides[current].dataset.noteSeconds}초, 본편 ${duration(mainSeconds)} (부록은 질의응답용)`;
    document.getElementById("notes-text").textContent = slides[current].querySelector(".speaker-notes").textContent;
    document.title = `${current + 1}. ${slides[current].querySelector("h1, h2").textContent} | Ant 실험`;
  }
  function fromHash() {
    const match = /^#([1-9]\d*)$/.exec(window.location.hash);
    const index = match ? Number(match[1]) - 1 : 0;
    show(Number.isSafeInteger(index) ? index : 0);
    if (window.location.hash !== `#${current + 1}`) history.replaceState(null, "", `#${current + 1}`);
  }
  function navigate(index) {
    const target = Math.max(0, Math.min(slides.length - 1, index));
    if (window.location.hash !== `#${target + 1}`) window.location.hash = String(target + 1);
  }
  function toggleNotes() {
    notesOpen = !notesOpen;
    document.body.classList.toggle("notes-open", notesOpen);
    notesPanel.hidden = !notesOpen;
    notesToggle.setAttribute("aria-pressed", String(notesOpen));
    fit();
  }
  previous.addEventListener("click", () => navigate(current - 1));
  next.addEventListener("click", () => navigate(current + 1));
  notesToggle.addEventListener("click", toggleNotes);
  window.addEventListener("keydown", event => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest("input, textarea, select, [contenteditable='true']")) return;
    if (event.key.toLowerCase() === "n") { event.preventDefault(); toggleNotes(); return; }
    // Preserve native video/button keyboard controls when they have focus.
    if (event.target.closest("video") && [" ", "ArrowLeft", "ArrowRight"].includes(event.key)) return;
    if (event.target.closest("button, a") && event.key === " ") return;
    const keys = {ArrowLeft: current - 1, PageUp: current - 1, ArrowRight: current + 1, PageDown: current + 1, " ": current + 1, Home: 0, End: slides.length - 1};
    if (Object.hasOwn(keys, event.key)) { event.preventDefault(); navigate(keys[event.key]); }
  });
  window.addEventListener("hashchange", fromHash);
  window.addEventListener("resize", fit);
  window.addEventListener("beforeprint", () => slides.forEach(slide => slide.removeAttribute("aria-hidden")));
  window.addEventListener("afterprint", () => show(current));
  fromHash();
  fit();
  document.documentElement.dataset.ready = "true";
})();
