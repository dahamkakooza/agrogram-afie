"""
Disease diagnosis model loader and inference wrapper.

Loads AgriYouthNet-Context+ TFLite model at AFIE startup.
Provides a singleton `disease_model` object with:
  - load()            — called from FastAPI lifespan
  - predict(img, ctx) — on-device inference fallback
  - get_metadata()    — model version, accuracy, class names
  - is_loaded         — bool flag

Model file expected at: <project_root>/models/agriyouthnet_context_plus_int8.tflite
Metadata JSON:          <project_root>/models/agriyouthnet_context_plus_int8.json
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.config.logging_config import get_logger

logger = get_logger("disease_model")


# ─── Locate project root (this file is at src/ml/disease_model.py) ─
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # → D:\agrogram-afie
_DEFAULT_MODEL = _PROJECT_ROOT / "models" / "agriyouthnet_context_plus_int8.tflite"
_DEFAULT_META  = _PROJECT_ROOT / "models" / "agriyouthnet_context_plus_int8.json"


class DiseaseModelManager:
    """
    Singleton wrapper around the TFLite interpreter.
    Lazy-loads on first `load()` call.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL
        self.metadata_path = Path(metadata_path) if metadata_path else _DEFAULT_META
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self._img_input_idx = None
        self._ctx_input_idx = None
        self._probs_output_idx = None
        self._attn_output_idx = None
        self._loaded = False
        self.metadata: Dict[str, Any] = {}
        self.class_names = []
        self.num_classes = 0
        self.context_dim = 4

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_version(self) -> str:
        return self.metadata.get("version", "unknown")

    def load(self) -> bool:
        """Load TFLite model + metadata. Returns True on success."""
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False

            # Import tensorflow lazily — only needed if disease feature is on
            import tensorflow as tf

            logger.info(f"Loading TFLite model: {self.model_path.name}")
            self.interpreter = tf.lite.Interpreter(
                model_path=str(self.model_path),
                num_threads=4,
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()

            # Locate inputs/outputs by shape (order is not guaranteed)
            for inp in self.input_details:
                shape = inp["shape"]
                if len(shape) == 4 and shape[-1] == 3:
                    self._img_input_idx = inp["index"]
                elif len(shape) == 2 and shape[-1] == self.context_dim:
                    self._ctx_input_idx = inp["index"]

            for out in self.output_details:
                shape = out["shape"]
                if shape[-1] == 10:  # num_classes
                    self._probs_output_idx = out["index"]
                elif shape[-1] == self.context_dim:
                    self._attn_output_idx = out["index"]

            if any(i is None for i in (
                self._img_input_idx, self._ctx_input_idx,
                self._probs_output_idx, self._attn_output_idx,
            )):
                logger.error(
                    f"Could not locate all tensors. "
                    f"inputs={[(i['name'], i['shape'].tolist()) for i in self.input_details]} "
                    f"outputs={[(o['name'], o['shape'].tolist()) for o in self.output_details]}"
                )
                return False

            # Load metadata JSON if present
            if self.metadata_path.exists():
                with open(self.metadata_path) as f:
                    self.metadata = json.load(f)
                self.class_names = self.metadata.get("class_names", [])
                self.num_classes = self.metadata.get("num_classes", 10)
                self.context_dim = self.metadata.get("context_dim", 4)
            else:
                logger.warning(f"Metadata JSON not found: {self.metadata_path}")

            self._loaded = True
            logger.info(
                f"✓ Disease model loaded: v{self.model_version}, "
                f"{self.num_classes} classes, "
                f"size={self.model_path.stat().st_size / 1024:.1f} KB"
            )
            return True

        except Exception as e:
            logger.error(f"Disease model load failed: {type(e).__name__}: {e}")
            return False

    def predict(
        self,
        image: np.ndarray,          # (224, 224, 3) float32 in [0,1]
        context: np.ndarray,        # (4,) float32 one-hot crop type
    ) -> Dict[str, Any]:
        """
        Run inference. Returns dict with:
          predicted_class:   int 0-9
          predicted_label:   str
          confidence:        float
          probabilities:     list[float] (length 10)
          attention_weights: list[float] (length 4)
        """
        if not self._loaded:
            raise RuntimeError("Disease model not loaded")

        img_in = image[None, ...].astype(np.float32)   # (1, 224, 224, 3)
        ctx_in = context[None, :].astype(np.float32)   # (1, 4)

        self.interpreter.set_tensor(self._img_input_idx, img_in)
        self.interpreter.set_tensor(self._ctx_input_idx, ctx_in)
        self.interpreter.invoke()

        probs = self.interpreter.get_tensor(self._probs_output_idx)[0]
        attn = self.interpreter.get_tensor(self._attn_output_idx)[0]

        pred_idx = int(np.argmax(probs))
        label = (
            self.class_names[pred_idx]
            if pred_idx < len(self.class_names)
            else f"class_{pred_idx}"
        )

        return {
            "predicted_class": pred_idx,
            "predicted_label": label,
            "confidence": float(probs[pred_idx]),
            "probabilities": probs.tolist(),
            "attention_weights": attn.tolist(),
        }

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "loaded": self._loaded,
            "path": str(self.model_path),
            "exists": self.model_path.exists(),
            "size_kb": (
                round(self.model_path.stat().st_size / 1024, 2)
                if self.model_path.exists() else None
            ),
            **self.metadata,
        }

    def release(self):
        """Release the interpreter (called at shutdown)."""
        self.interpreter = None
        self._loaded = False


# ─── Module-level singleton ─────────────────────────────────────
disease_model = DiseaseModelManager()