"""Liveness model adapter."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from modules.livenessnet import LivenessNet
from modules.model_integrity import verify_sha256


class LivenessDetector:
    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        real_class: int = 1,
        real_threshold: float = 0.5,
        model_sha256: str | None = None,
        require_model_sha256: bool = False,
    ):
        self.device = self._resolve_device(device)
        self.real_class = real_class
        self.real_threshold = real_threshold
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy liveness model: {model_path}")
        verify_sha256(
            model_path,
            model_sha256,
            required=require_model_sha256,
        )

        self.model = LivenessNet(width=32, height=32, depth=3, classes=2)
        try:
            checkpoint = torch.load(
                model_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError as exc:
            raise RuntimeError(
                "PyTorch quá cũ; cần phiên bản hỗ trợ torch.load(weights_only=True)"
            ) from exc
        if not isinstance(checkpoint, dict):
            raise ValueError("Liveness checkpoint phải là state dictionary")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            raise ValueError("Liveness checkpoint không chứa state dictionary hợp lệ")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        normalized = (device or "auto").lower()
        if normalized == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DEVICE=cuda nhưng máy không có CUDA khả dụng")
        if normalized not in {"cpu", "cuda"}:
            raise ValueError("DEVICE phải là auto, cpu hoặc cuda")
        return normalized

    def is_real(self, face_img: np.ndarray) -> bool:
        image = cv2.resize(face_img, (32, 32), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            output = self.model(tensor)
            probability = float(output[0, self.real_class].item())
            prediction = int(torch.argmax(output, dim=1).item())
        return prediction == self.real_class and probability >= self.real_threshold
