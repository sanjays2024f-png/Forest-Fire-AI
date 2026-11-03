from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
LABELS = ["fire", "smoke", "no_fire"]
class EfficientNetClassifier:
    """EfficientNet-B0 wrapper. Honest: unavailable unless real weights load."""
    def __init__(self, model_path: str | None = None):
        self.model_path = str(model_path or "")
        self.loaded = False
        self._model = None
        self._prep = None
        if self.model_path and Path(self.model_path).is_file():
            try:
                import torch
                from torchvision import models
                m = models.efficientnet_b0()
                m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, 3)
                m.load_state_dict(torch.load(self.model_path, map_location="cpu"))
                m.eval()
                self._model = m
                self.loaded = True
            except Exception:
                self._model = None
                self.loaded = False
    @property
    def available(self) -> bool:
        return self._model is not None
    def predict_image(self, image_path: str) -> Dict[str, Any] | None:
        if self._model is None:
            return None
        try:
            import torch
            from PIL import Image
            from torchvision import transforms
            prep = transforms.Compose([
                transforms.Resize((224, 224)), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            with Image.open(image_path) as im:
                t = prep(im.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(self._model(t), dim=1)[0].tolist()
            scores = {LABELS[i]: round(float(probs[i]), 4) for i in range(3)}
            label = max(scores, key=scores.get)
            return {"label": label, "confidence": scores[label],
                    "probabilities": scores, "backend": "efficientnet-b0"}
        except Exception:
            return None
    def classify(self, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return []

