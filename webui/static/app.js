const modelSelect = document.getElementById("model-select");
const imageInput = document.getElementById("image-input");
const fileNameText = document.getElementById("file-name");
const previewImage = document.getElementById("preview-image");
const previewCanvas = document.getElementById("preview-canvas");
const emptyPreview = document.getElementById("empty-preview");
const detectForm = document.getElementById("detect-form");
const submitButton = document.getElementById("submit-button");
const statusPill = document.getElementById("status-pill");
const resultEmpty = document.getElementById("result-empty");
const resultContent = document.getElementById("result-content");

const predictionText = document.getElementById("prediction-text");
const modelText = document.getElementById("model-text");
const stegoProbability = document.getElementById("stego-probability");
const coverProbability = document.getElementById("cover-probability");
const stegoBar = document.getElementById("stego-bar");
const coverBar = document.getElementById("cover-bar");
const deviceText = document.getElementById("device-text");
const confidenceText = document.getElementById("confidence-text");
const preprocessText = document.getElementById("preprocess-text");

function setStatus(label, tone) {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${tone}`;
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();

    modelSelect.innerHTML = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "请选择检测模型";
    modelSelect.appendChild(placeholder);

    data.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.name;
      modelSelect.appendChild(option);
    });
  } catch (error) {
    modelSelect.innerHTML = '<option value="">模型加载失败</option>';
    setStatus("模型加载失败", "warning");
  }
}

function resetPreview() {
  previewImage.style.display = "none";
  previewCanvas.style.display = "none";
  emptyPreview.style.display = "block";
}

function drawImageToCanvas(width, height, rgbaData) {
  const context = previewCanvas.getContext("2d");
  previewCanvas.width = width;
  previewCanvas.height = height;
  const imageData = new ImageData(rgbaData, width, height);
  context.putImageData(imageData, 0, 0);
  previewCanvas.style.display = "block";
  emptyPreview.style.display = "none";
}

function renderPgmPreview(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  let index = 0;

  function isWhitespace(value) {
    return value === 9 || value === 10 || value === 13 || value === 32;
  }

  function skipWhitespaceAndComments() {
    while (index < bytes.length) {
      if (bytes[index] === 35) {
        while (index < bytes.length && bytes[index] !== 10) {
          index += 1;
        }
      } else if (isWhitespace(bytes[index])) {
        index += 1;
      } else {
        break;
      }
    }
  }

  function readToken() {
    skipWhitespaceAndComments();
    const start = index;
    while (index < bytes.length && !isWhitespace(bytes[index]) && bytes[index] !== 35) {
      index += 1;
    }
    return new TextDecoder().decode(bytes.slice(start, index));
  }

  const magic = readToken();
  const width = Number.parseInt(readToken(), 10);
  const height = Number.parseInt(readToken(), 10);
  const maxValue = Number.parseInt(readToken(), 10);

  if (!["P2", "P5"].includes(magic) || !width || !height || !maxValue) {
    throw new Error("不支持的 PGM 文件格式");
  }

  skipWhitespaceAndComments();
  const pixelCount = width * height;
  const rgba = new Uint8ClampedArray(pixelCount * 4);

  if (magic === "P5") {
    const pixelBytes = bytes.slice(index, index + pixelCount);
    if (pixelBytes.length < pixelCount) {
      throw new Error("PGM 文件数据不完整");
    }
    for (let i = 0; i < pixelCount; i += 1) {
      const value = Math.round((pixelBytes[i] / maxValue) * 255);
      const offset = i * 4;
      rgba[offset] = value;
      rgba[offset + 1] = value;
      rgba[offset + 2] = value;
      rgba[offset + 3] = 255;
    }
  } else {
    for (let i = 0; i < pixelCount; i += 1) {
      const token = readToken();
      if (!token) {
        throw new Error("PGM 文件数据不完整");
      }
      const value = Math.round((Number.parseInt(token, 10) / maxValue) * 255);
      const offset = i * 4;
      rgba[offset] = value;
      rgba[offset + 1] = value;
      rgba[offset + 2] = value;
      rgba[offset + 3] = 255;
    }
  }

  drawImageToCanvas(width, height, rgba);
}

function previewSelectedImage(file) {
  if (!file) {
    resetPreview();
    fileNameText.textContent = "尚未选择图片";
    return;
  }

  fileNameText.textContent = file.name;
  const isPgm = file.name.toLowerCase().endsWith(".pgm");

  if (isPgm) {
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        previewImage.style.display = "none";
        renderPgmPreview(event.target.result);
      } catch (error) {
        resetPreview();
        emptyPreview.textContent = error.message;
      }
    };
    reader.readAsArrayBuffer(file);
    return;
  }

  const reader = new FileReader();
  reader.onload = (event) => {
    previewCanvas.style.display = "none";
    previewImage.src = event.target.result;
    previewImage.style.display = "block";
    emptyPreview.style.display = "none";
  };
  reader.readAsDataURL(file);
}

function showResult(data) {
  resultEmpty.classList.add("hidden");
  resultContent.classList.remove("hidden");

  const stegoPercent = (data.stego_probability * 100).toFixed(2);
  const coverPercent = (data.cover_probability * 100).toFixed(2);
  const confidencePercent = (data.confidence * 100).toFixed(2);
  const predictionLabel = data.prediction === "stego"
    ? "疑似隐写图像"
    : "疑似原始载体图像";

  predictionText.textContent = predictionLabel;
  modelText.textContent = `模型：${data.model.name}`;
  stegoProbability.textContent = `${stegoPercent}%`;
  coverProbability.textContent = `${coverPercent}%`;
  stegoBar.style.width = `${stegoPercent}%`;
  coverBar.style.width = `${coverPercent}%`;
  deviceText.textContent = data.device;
  confidenceText.textContent = `${confidencePercent}%`;
  preprocessText.textContent = `${data.preprocess.mode}, ${data.preprocess.resize}`;

  if (data.prediction === "stego") {
    setStatus("疑似隐写", "warning");
  } else {
    setStatus("疑似载体", "success");
  }
}

detectForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = imageInput.files[0];
  if (!modelSelect.value) {
    setStatus("请选择模型", "warning");
    return;
  }
  if (!file) {
    setStatus("请上传图片", "warning");
    return;
  }

  submitButton.disabled = true;
  submitButton.textContent = "检测中...";
  setStatus("正在推理", "loading");

  const formData = new FormData();
  formData.append("model_id", modelSelect.value);
  formData.append("image", file);

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "检测失败");
    }

    showResult(data);
  } catch (error) {
    resultContent.classList.add("hidden");
    resultEmpty.classList.remove("hidden");
    resultEmpty.textContent = error.message;
    setStatus("检测失败", "warning");
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "执行检测";
  }
});

imageInput.addEventListener("change", () => {
  previewSelectedImage(imageInput.files[0]);
});

loadModels();
