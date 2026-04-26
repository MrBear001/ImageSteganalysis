const overviewPill = document.getElementById("overview-pill");
const overviewCards = document.getElementById("overview-cards");
const accuracyBars = document.getElementById("accuracy-bars");
const aucBars = document.getElementById("auc-bars");
const metricsTbody = document.getElementById("metrics-tbody");
const algorithmFilters = document.getElementById("algorithm-filters");
const globalImages = document.getElementById("global-images");
const experimentImages = document.getElementById("experiment-images");

let allRows = [];
let activeAlgorithm = "ALL";

function setOverviewStatus(text, tone) {
  overviewPill.textContent = text;
  overviewPill.className = `status-pill ${tone}`;
}

function toPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  const normalized = value > 1 ? value : value * 100;
  return `${normalized.toFixed(2)}%`;
}

function toFixedNumber(value, digits = 4) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function getFilteredRows() {
  if (activeAlgorithm === "ALL") {
    return allRows;
  }
  return allRows.filter((item) => item.algorithm === activeAlgorithm);
}

function drawBars(container, rows, valueKey, label) {
  container.innerHTML = "";
  const validRows = rows.filter((item) => typeof item[valueKey] === "number");
  if (!validRows.length) {
    container.innerHTML = '<div class="result-empty">暂无可展示数据</div>';
    return;
  }

  const maxValue = Math.max(...validRows.map((item) => item[valueKey]));
  validRows.forEach((item) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const width = maxValue > 0 ? (item[valueKey] / maxValue) * 100 : 0;
    row.innerHTML = `
      <div class="bar-label">${item.algorithm} ${item.embedding_rate} bpp</div>
      <div class="bar-track">
        <div class="bar-fill" style="width:${width.toFixed(2)}%"></div>
      </div>
      <div class="bar-value">${label === "percent" ? toPercent(item[valueKey]) : toFixedNumber(item[valueKey], 4)}</div>
    `;
    container.appendChild(row);
  });
}

function renderOverview(rows) {
  if (!rows.length) {
    overviewCards.innerHTML = '<div class="result-empty">未发现实验结果文件。</div>';
    return;
  }

  const sortedByAcc = [...rows].sort((a, b) => (b.final_test_accuracy || 0) - (a.final_test_accuracy || 0));
  const sortedByAuc = [...rows].sort((a, b) => (b.auc || 0) - (a.auc || 0));

  const bestAcc = sortedByAcc[0];
  const bestAuc = sortedByAuc[0];
  const avgAcc = rows.reduce((sum, item) => sum + (item.final_test_accuracy || 0), 0) / rows.length;
  const avgAuc = rows.reduce((sum, item) => sum + (item.auc || 0), 0) / rows.length;

  overviewCards.innerHTML = `
    <article class="overview-card">
      <span>最佳 Final Test Acc</span>
      <strong>${toPercent(bestAcc.final_test_accuracy)}</strong>
      <p>${bestAcc.algorithm} ${bestAcc.embedding_rate} bpp</p>
    </article>
    <article class="overview-card">
      <span>最佳 AUC</span>
      <strong>${toFixedNumber(bestAuc.auc, 4)}</strong>
      <p>${bestAuc.algorithm} ${bestAuc.embedding_rate} bpp</p>
    </article>
    <article class="overview-card">
      <span>平均 Final Test Acc</span>
      <strong>${toPercent(avgAcc)}</strong>
      <p>${rows.length} 组实验</p>
    </article>
    <article class="overview-card">
      <span>平均 AUC</span>
      <strong>${toFixedNumber(avgAuc, 4)}</strong>
      <p>${activeAlgorithm === "ALL" ? "覆盖全部算法" : `${activeAlgorithm} 子集`}</p>
    </article>
  `;
}

function renderTable(rows) {
  metricsTbody.innerHTML = "";
  rows.forEach((item) => {
    const tr = document.createElement("tr");
    const conf = item.confusion_matrix || {};
    const confText = `${conf.tn ?? "-"} / ${conf.fp ?? "-"} / ${conf.fn ?? "-"} / ${conf.tp ?? "-"}`;
    tr.innerHTML = `
      <td>${item.algorithm} ${item.embedding_rate} bpp</td>
      <td>${item.epochs || "-"}</td>
      <td>${toPercent(item.accuracy)}</td>
      <td>${toPercent(item.precision)}</td>
      <td>${toPercent(item.recall)}</td>
      <td>${toPercent(item.f1_score)}</td>
      <td>${toFixedNumber(item.auc, 4)}</td>
      <td>${toFixedNumber(item.average_loss, 4)}</td>
      <td>${confText}</td>
    `;
    metricsTbody.appendChild(tr);
  });
}

function renderFilters(rows) {
  const algorithms = Array.from(new Set(rows.map((item) => item.algorithm))).sort();
  const allOptions = ["ALL", ...algorithms];
  algorithmFilters.innerHTML = "";

  allOptions.forEach((algorithm) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `chip ${activeAlgorithm === algorithm ? "active" : ""}`;
    button.textContent = algorithm === "ALL" ? "全部算法" : algorithm;
    button.addEventListener("click", () => {
      activeAlgorithm = algorithm;
      renderFilters(allRows);
      renderPage();
    });
    algorithmFilters.appendChild(button);
  });
}

function createImageCard(title, imagePath, subtitle = "") {
  const card = document.createElement("article");
  card.className = "image-card";
  card.innerHTML = `
    <h3>${title}</h3>
    ${subtitle ? `<p>${subtitle}</p>` : ""}
    <img src="${imagePath}" alt="${title}">
  `;
  return card;
}

function renderGlobalImages() {
  globalImages.innerHTML = "";
  globalImages.appendChild(
    createImageCard(
      "整体实验对比图",
      "/evaluation_outputs/log_plots/experiment_comparison.png",
      "展示不同实验配置下的综合性能对比"
    )
  );
}

function renderExperimentImages(rows) {
  experimentImages.innerHTML = "";
  if (!rows.length) {
    experimentImages.innerHTML = '<div class="result-empty">当前筛选下暂无结果图。</div>';
    return;
  }

  rows.forEach((item) => {
    const baseTitle = `${item.algorithm} ${item.embedding_rate} bpp`;
    const images = item.images || {};
    experimentImages.appendChild(
      createImageCard(`${baseTitle} - ROC 曲线`, images.roc_curve, "模型区分能力与AUC可视化")
    );
    experimentImages.appendChild(
      createImageCard(`${baseTitle} - 混淆矩阵`, images.confusion_matrix, "分类结果分布")
    );
    experimentImages.appendChild(
      createImageCard(`${baseTitle} - 概率直方图`, images.probability_histogram, "载体/隐写概率分布")
    );
  });
}

function renderPage() {
  const rows = getFilteredRows();
  renderOverview(rows);
  drawBars(accuracyBars, rows, "final_test_accuracy", "percent");
  drawBars(aucBars, rows, "auc", "number");
  renderTable(rows);
  renderExperimentImages(rows);
}

async function loadExperiments() {
  try {
    setOverviewStatus("正在加载", "loading");
    const response = await fetch("/api/experiments");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "实验结果加载失败");
    }

    allRows = data.experiments || [];
    renderFilters(allRows);
    renderGlobalImages();
    renderPage();
    setOverviewStatus("加载完成", "success");
  } catch (error) {
    setOverviewStatus("加载失败", "warning");
    overviewCards.innerHTML = `<div class="result-empty">${error.message}</div>`;
    accuracyBars.innerHTML = '<div class="result-empty">无法绘制图表</div>';
    aucBars.innerHTML = '<div class="result-empty">无法绘制图表</div>';
    metricsTbody.innerHTML = "";
    algorithmFilters.innerHTML = "";
    globalImages.innerHTML = "";
    experimentImages.innerHTML = "";
  }
}

loadExperiments();
