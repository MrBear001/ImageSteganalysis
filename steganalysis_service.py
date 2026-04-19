import os
import re
from dataclasses import dataclass
from io import BytesIO
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from LWENet import lwenet


MODEL_LABELS = {
    "suni": "S-UNIWARD",
    "wow": "WOW",
}

CHECKPOINT_PATTERN = re.compile(r"lwenet_epoch_(\d+)\.pkl$", re.IGNORECASE)
DIR_PATTERN = re.compile(r"checkpoints-(suni|wow)(\d+(?:\.\d+)?)$", re.IGNORECASE)


@dataclass
class ModelInfo:
    model_id: str
    algorithm: str
    embedding_rate: str
    checkpoint_dir: str
    weights_path: str
    epoch: int
    display_name: str


class ModelRegistry:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self._models = self._discover_models()

    def _discover_models(self) -> Dict[str, ModelInfo]:
        models: Dict[str, ModelInfo] = {}
        for entry in sorted(os.listdir(self.root_dir)):
            full_path = os.path.join(self.root_dir, entry)
            if not os.path.isdir(full_path):
                continue

            dir_match = DIR_PATTERN.match(entry)
            if not dir_match:
                continue

            algorithm_key = dir_match.group(1).lower()
            embedding_rate = dir_match.group(2)
            weights_path, epoch = self._find_latest_checkpoint(full_path)
            if not weights_path:
                continue

            model_id = f"{algorithm_key}_{embedding_rate.replace('.', '_')}"
            algorithm = MODEL_LABELS.get(algorithm_key, algorithm_key.upper())
            display_name = f"{algorithm} {embedding_rate} bpp"

            models[model_id] = ModelInfo(
                model_id=model_id,
                algorithm=algorithm,
                embedding_rate=embedding_rate,
                checkpoint_dir=full_path,
                weights_path=weights_path,
                epoch=epoch,
                display_name=display_name,
            )

        return models

    def _find_latest_checkpoint(self, checkpoint_dir: str):
        best_epoch = -1
        best_path = None
        for filename in os.listdir(checkpoint_dir):
            match = CHECKPOINT_PATTERN.match(filename)
            if not match:
                continue
            epoch = int(match.group(1))
            if epoch > best_epoch:
                best_epoch = epoch
                best_path = os.path.join(checkpoint_dir, filename)
        return best_path, best_epoch

    def list_models(self) -> List[ModelInfo]:
        return [self._models[key] for key in sorted(self._models.keys())]

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        return self._models.get(model_id)


class SteganalysisService:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.registry = ModelRegistry(self.root_dir)
        self._model_cache: Dict[str, torch.nn.Module] = {}

    def list_models(self) -> List[Dict[str, str]]:
        return [
            {
                "id": item.model_id,
                "name": item.display_name,
                "algorithm": item.algorithm,
                "embedding_rate": item.embedding_rate,
                "epoch": str(item.epoch),
                "weights_path": item.weights_path,
            }
            for item in self.registry.list_models()
        ]

    def _load_model(self, model_id: str) -> torch.nn.Module:
        if model_id in self._model_cache:
            return self._model_cache[model_id]

        model_info = self.registry.get_model(model_id)
        if model_info is None:
            raise ValueError(f"Unknown model id: {model_id}")

        model = lwenet().to(self.device)
        state_dict = torch.load(model_info.weights_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        self._model_cache[model_id] = model
        return model

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(BytesIO(image_bytes)).convert("L")
        image = image.resize((256, 256), Image.BICUBIC)
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(image_array)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, model_id: str, image_bytes: bytes) -> Dict[str, object]:
        model_info = self.registry.get_model(model_id)
        if model_info is None:
            raise ValueError("Selected model does not exist.")

        model = self._load_model(model_id)
        image_tensor = self.preprocess_image(image_bytes)

        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = F.softmax(logits, dim=1)[0]

        cover_probability = float(probabilities[0].item())
        stego_probability = float(probabilities[1].item())
        predicted_label = "stego" if stego_probability >= cover_probability else "cover"

        return {
            "model": {
                "id": model_info.model_id,
                "name": model_info.display_name,
                "algorithm": model_info.algorithm,
                "embedding_rate": model_info.embedding_rate,
                "epoch": model_info.epoch,
            },
            "device": str(self.device),
            "prediction": predicted_label,
            "stego_probability": stego_probability,
            "cover_probability": cover_probability,
            "confidence": max(stego_probability, cover_probability),
            "preprocess": {
                "mode": "grayscale",
                "resize": "256x256",
                "normalization": "pixel / 255.0",
            },
        }
