"""
Quantize AgriYouthNet-Context+ to TFLite INT8.

Pipeline:
  1. Load trained Keras model
  2. Build representative dataset from real val images
  3. Convert to TFLite F32
  4. Convert to TFLite INT8
  5. Verify parity on val set
  6. Save .tflite + metadata JSON

IMPORTANT: The model was trained on images in [0,1] range (no ImageNet
normalization). This script must use the SAME preprocessing for the
representative dataset and for parity evaluation.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf

from src.agriyouthnet_context_plus import (
    build_agriyouthnet_context_plus, MultiHeadCrossAttention, MeanOverHeads,
)
from src.dataset import PV_TO_INDEX, CROP_PER_CLASS, CROP_INDEX, CLASS_NAMES

# ─── Config ──────────────────────────────────────────────────────
ROOT = Path("data/plantvillage")
IMG_SIZE = 224
N_PER_CLASS = 500
SEED = 42
CONTEXT_DIM = 4
NUM_CLASSES = 10

MODEL_DIR = Path("training/models/context_plus_4h_noaug")
KERAS_PATH = MODEL_DIR / "best.keras"
OUT_DIR = Path("training/models/tflite")
OUT_DIR.mkdir(parents=True, exist_ok=True)
F32_PATH = OUT_DIR / "agriyouthnet_context_plus_f32.tflite"
INT8_PATH = OUT_DIR / "agriyouthnet_context_plus_int8.tflite"
META_PATH = OUT_DIR / "agriyouthnet_context_plus_int8.json"


# ─── Custom object registry (for loading Keras model with custom layers) ─
CUSTOM_OBJECTS = {
    "MultiHeadCrossAttention": MultiHeadCrossAttention,
    "MeanOverHeads": MeanOverHeads,
}


# ─── Data loading (matches train_numpy.py exactly) ──────────────
def load_val_data():
    """
    Load the same 500/class split used in training, return val only.
    Images are loaded at [0,1] range — matches what the model was trained on.
    """
    print(f"Loading images (max {N_PER_CLASS}/class)...")
    X_img, X_ctx, y = [], [], []
    for class_name, idx in PV_TO_INDEX.items():
        folder = ROOT / class_name
        files = (list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))
                 + list(folder.glob("*.PNG")) + list(folder.glob("*.png")))
        files = files[:N_PER_CLASS]
        crop_idx = CROP_INDEX[CROP_PER_CLASS[idx]]
        for f in files:
            img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            X_img.append(arr)
            ctx = np.zeros(CONTEXT_DIM, dtype=np.float32); ctx[crop_idx] = 1.0
            X_ctx.append(ctx)
            y.append(idx)

    X_img = np.stack(X_img)
    X_ctx = np.stack(X_ctx)
    y = np.array(y, dtype=np.int32)

    # Same split as training (stratified, seed 42)
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)

    # IMPORTANT: no ImageNet normalization. Model expects [0,1] input.
    X_img_val = X_img[va]
    return X_img_val, X_ctx[va], y[va]


# ─── Representative dataset for INT8 calibration ─────────────────
def make_rep_dataset(X_img_val, X_ctx_val, n_samples=100):
    """Yield (image, context) tuples. Real images → accurate calibration."""
    idxs = np.random.choice(len(X_img_val),
                            size=min(n_samples, len(X_img_val)),
                            replace=False)
    def gen():
        for i in idxs:
            yield [
                X_img_val[i:i+1].astype(np.float32),
                X_ctx_val[i:i+1].astype(np.float32),
            ]
    return gen


# ─── Evaluation helper ───────────────────────────────────────────
def evaluate_tflite(tflite_path, X_img_val, X_ctx_val, y_val, verbose=False):
    """Run TFLite inference on the full val set."""
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    # Identify inputs by shape
    img_input_idx = None
    ctx_input_idx = None
    for inp in inputs:
        shape = inp["shape"]
        if shape[-1] == 3 and len(shape) == 4:
            img_input_idx = inp["index"]
        elif shape[-1] == CONTEXT_DIM:
            ctx_input_idx = inp["index"]

    # Identify outputs by shape
    probs_idx = None
    attn_idx = None
    for out in outputs:
        shape = out["shape"]
        if shape[-1] == NUM_CLASSES:
            probs_idx = out["index"]
        elif shape[-1] == CONTEXT_DIM:
            attn_idx = out["index"]

    if verbose:
        print(f"  inputs:  image_idx={img_input_idx} ctx_idx={ctx_input_idx}")
        print(f"  outputs: probs_idx={probs_idx} attn_idx={attn_idx}")

    correct = 0
    total = 0
    probs_list, attn_list = [], []
    for i in range(len(X_img_val)):
        interpreter.set_tensor(img_input_idx, X_img_val[i:i+1].astype(np.float32))
        interpreter.set_tensor(ctx_input_idx, X_ctx_val[i:i+1].astype(np.float32))
        interpreter.invoke()
        probs = interpreter.get_tensor(probs_idx)[0]
        attn = interpreter.get_tensor(attn_idx)[0]
        probs_list.append(probs)
        attn_list.append(attn)
        pred = int(np.argmax(probs))
        if pred == y_val[i]:
            correct += 1
        total += 1

    acc = correct / total
    return acc, np.array(probs_list), np.array(attn_list)


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("QUANTIZATION PIPELINE")
    print("=" * 60)

    if not KERAS_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {KERAS_PATH}")

    # ─── 1. Load Keras model ─────────────────────────────────────
    print(f"\n[1/6] Loading Keras model from {KERAS_PATH}...")
    model = tf.keras.models.load_model(
        str(KERAS_PATH),
        custom_objects=CUSTOM_OBJECTS,
        compile=False,
        safe_mode=False,
    )
    n_params = model.count_params()
    print(f"  params: {n_params:,}")
    print(f"  inputs:  {[i.name for i in model.inputs]}")
    print(f"  outputs: {[o.name for o in model.outputs]}")

    # ─── 2. Load val data ────────────────────────────────────────
    print("\n[2/6] Loading validation data...")
    X_img_val, X_ctx_val, y_val = load_val_data()
    print(f"  val set: {len(y_val)} samples")
    print(f"  image range: [{X_img_val.min():.3f}, {X_img_val.max():.3f}]  "
          f"mean={X_img_val.mean():.3f}")

    # ─── 3. Keras baseline on val ────────────────────────────────
    print("\n[3/6] Keras baseline accuracy...")
    t0 = time.time()
    keras_preds = model.predict(
        {"leaf_image": X_img_val, "context_vector": X_ctx_val},
        batch_size=64, verbose=0,
    )
    keras_probs = keras_preds[0] if isinstance(keras_preds, list) else keras_preds
    keras_acc = float((keras_probs.argmax(axis=1) == y_val).mean())
    print(f"  Keras val accuracy: {keras_acc:.4f}  ({time.time()-t0:.1f}s)")

    # ─── 4. TFLite F32 conversion ────────────────────────────────
    print("\n[4/6] Converting to TFLite F32...")
    t0 = time.time()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_f32 = converter.convert()
    F32_PATH.write_bytes(tflite_f32)
    print(f"  ✓ {F32_PATH}  ({len(tflite_f32)/1024:.2f} KB, {time.time()-t0:.1f}s)")

    # ─── 5. TFLite INT8 conversion ───────────────────────────────
    print("\n[5/6] Converting to TFLite INT8...")
    t0 = time.time()
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_rep_dataset(
        X_img_val, X_ctx_val, n_samples=100)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_int8 = converter.convert()
    INT8_PATH.write_bytes(tflite_int8)
    print(f"  ✓ {INT8_PATH}  ({len(tflite_int8)/1024:.2f} KB, {time.time()-t0:.1f}s)")

    # ─── 6. Verify parity ────────────────────────────────────────
    print("\n[6/6] Verifying parity on val set...")

    print("\n  Evaluating F32 TFLite...")
    f32_acc, f32_probs, _ = evaluate_tflite(
        F32_PATH, X_img_val, X_ctx_val, y_val, verbose=True)
    print(f"  F32 TFLite accuracy: {f32_acc:.4f}")

    print("\n  Evaluating INT8 TFLite...")
    int8_acc, int8_probs, int8_attn = evaluate_tflite(
        INT8_PATH, X_img_val, X_ctx_val, y_val, verbose=True)
    print(f"  INT8 TFLite accuracy: {int8_acc:.4f}")

    # Delta from Keras
    f32_delta = keras_acc - f32_acc
    int8_delta = keras_acc - int8_acc
    print(f"\n  F32 drop vs Keras:  {f32_delta*100:+.2f} pp")
    print(f"  INT8 drop vs Keras: {int8_delta*100:+.2f} pp")

    # ─── Metadata JSON ───────────────────────────────────────────
    metadata = {
        "model": "agriyouthnet_context_plus",
        "version": "1.0.0",
        "params": int(n_params),
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "context_dim": CONTEXT_DIM,
        "crop_index": CROP_INDEX,
        "input_size": IMG_SIZE,
        "preprocessing": {
            "range": "[0,1]",
            "steps": [
                "PIL: Image.open(path).convert('RGB').resize((224,224))",
                "numpy: np.array(img).astype(np.float32) / 255.0",
                "NO further normalization",
            ],
        },
        "inputs": [
            {"name": "leaf_image", "shape": [1, 224, 224, 3], "dtype": "float32"},
            {"name": "context_vector", "shape": [1, 4], "dtype": "float32"},
        ],
        "outputs": [
            {"name": "class_probs", "shape": [1, 10], "dtype": "float32"},
            {"name": "attention_weights", "shape": [1, 4], "dtype": "float32"},
        ],
        "quantization": "INT8 (post-training, representative-dataset calibration)",
        "files": {
            "f32": F32_PATH.name,
            "int8": INT8_PATH.name,
        },
        "sizes_kb": {
            "f32": round(len(tflite_f32) / 1024, 2),
            "int8": round(len(tflite_int8) / 1024, 2),
        },
        "accuracy": {
            "keras_val": round(keras_acc, 4),
            "f32_val": round(f32_acc, 4),
            "int8_val": round(int8_acc, 4),
            "f32_drop_pp": round(f32_delta * 100, 2),
            "int8_drop_pp": round(int8_delta * 100, 2),
        },
        "targets": {
            "size_under_kb": 200,
            "accuracy_drop_under_pp": 1.0,
        },
        "training": {
            "n_per_class": N_PER_CLASS,
            "batch_size": 64,
            "epochs_run": 35,
            "augmentation": "off",
        },
    }
    META_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\n✓ Metadata saved: {META_PATH}")

    # ─── Verdict ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    size_ok = len(tflite_int8) / 1024 < 200
    acc_ok = int8_delta * 100 < 1.0
    print(f"  INT8 size:           {len(tflite_int8)/1024:.2f} KB  "
          f"{'✅' if size_ok else '❌'}  (target < 200 KB)")
    print(f"  INT8 accuracy drop:  {int8_delta*100:+.2f} pp  "
          f"{'✅' if acc_ok else '❌'}  (target < 1.0 pp)")
    print(f"  INT8 val accuracy:   {int8_acc:.4f}")

    if size_ok and acc_ok:
        print("\n  🎉 BOTH TARGETS MET — ready for deployment")
    elif size_ok and not acc_ok:
        print("\n  ⚠️  Size OK but accuracy dropped too much")
    elif acc_ok and not size_ok:
        print("\n  ⚠️  Accuracy OK but model too large")
    else:
        print("\n  ❌ Both targets missed")

    return metadata


if __name__ == "__main__":
    metadata = main()