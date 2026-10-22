"""
Per-class evaluation: confusion matrices, per-class F1,
early-vs-late blight pair accuracy for H1 analysis.

Loads each trained model from disk, evaluates on the same 930-sample
validation split used during training, and reports:
  - Aggregate accuracy
  - Macro F1
  - Per-class F1
  - Tomato early-vs-late blight pair accuracy
  - Potato early-vs-late blight pair accuracy
  - Full confusion matrix

Outputs: training/logs/per_class_analysis.json
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
from pathlib import Path

# ─── Path bootstrap ──────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent.parent     # D:\agrogram-afie\training
sys.path.insert(0, str(_SCRIPT_DIR))
# ─────────────────────────────────────────────────────────────────

import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from src.dataset import PV_TO_INDEX, CROP_PER_CLASS, CROP_INDEX, CLASS_NAMES
from src.agriyouthnet_context_plus import (
    build_agriyouthnet_context_plus, MultiHeadCrossAttention, MeanOverHeads,
)


# ─── Config ──────────────────────────────────────────────────────
DATA_ROOT = _SCRIPT_DIR / "data" / "plantvillage"
MODELS_DIR = _SCRIPT_DIR / "models"
LOGS_DIR = _SCRIPT_DIR / "logs"

IMG_SIZE = 224
N_PER_CLASS = 500
SEED = 42
CONTEXT_DIM = 4
NUM_CLASSES = 10

CUSTOM_OBJECTS = {
    "MultiHeadCrossAttention": MultiHeadCrossAttention,
    "MeanOverHeads": MeanOverHeads,
}

MODELS_TO_EVAL = [
    ("context_plus_4h",        MODELS_DIR / "context_plus_4h_noaug"       / "best.keras"),
    ("context_concat",         MODELS_DIR / "context_concat_noaug"        / "best.keras"),
    ("context_gating",         MODELS_DIR / "context_gating_noaug"        / "best.keras"),
    ("context_plus_1h",        MODELS_DIR / "context_plus_1h_noaug"       / "best.keras"),
    ("context_plus_8h",        MODELS_DIR / "context_plus_8h_noaug"       / "best.keras"),
    ("context_plus_4h_narrow", MODELS_DIR / "context_plus_4h_narrow_noaug"/ "best.keras"),
]


# ─── Load validation set (same split as training) ───────────────
def load_val():
    print(f"Loading images from {DATA_ROOT}...")
    X_img, X_ctx, y = [], [], []
    for class_name, idx in PV_TO_INDEX.items():
        folder = DATA_ROOT / class_name
        if not folder.exists():
            print(f"  ⚠️  missing: {folder}")
            continue
        files = (list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))
                 + list(folder.glob("*.PNG")) + list(folder.glob("*.png")))
        files = files[:N_PER_CLASS]
        crop_idx = CROP_INDEX[CROP_PER_CLASS[idx]]
        for f in files:
            img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            X_img.append(np.array(img, dtype=np.float32) / 255.0)
            ctx = np.zeros(CONTEXT_DIM, dtype=np.float32); ctx[crop_idx] = 1.0
            X_ctx.append(ctx)
            y.append(idx)

    X_img = np.stack(X_img)
    X_ctx = np.stack(X_ctx)
    y = np.array(y, dtype=np.int32)
    print(f"  total: {len(y)} samples")
    print(f"  class dist: {np.bincount(y, minlength=NUM_CLASSES)}")

    # Same split as training
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y,
                              random_state=SEED)
    print(f"  val set: {len(va)} samples")
    return X_img[va], X_ctx[va], y[va]


# ─── Pair accuracy helper ───────────────────────────────────────
def pair_confusion(y_true, y_pred, cls_a, cls_b):
    """
    Accuracy restricted to samples that belong to cls_a or cls_b.
    Returns (correct, total, accuracy).
    """
    mask = (y_true == cls_a) | (y_true == cls_b)
    if mask.sum() == 0:
        return 0, 0, 0.0
    correct = int((y_pred[mask] == y_true[mask]).sum())
    total = int(mask.sum())
    return correct, total, correct / total


# ─── Main ────────────────────────────────────────────────────────
def main():
    X_img, X_ctx, y = load_val()

    results = {}
    for name, path in MODELS_TO_EVAL:
        print(f"\n{'='*60}")
        print(f"{name}")
        print(f"{'='*60}")

        if not path.exists():
            print(f"  ⚠️  not found: {path}")
            continue

        try:
            model = tf.keras.models.load_model(
                str(path),
                custom_objects=CUSTOM_OBJECTS,
                compile=False,
                safe_mode=False,
            )
        except Exception as e:
            print(f"  ❌ load failed: {type(e).__name__}: {e}")
            continue

        preds = model.predict(
            {"leaf_image": X_img, "context_vector": X_ctx},
            batch_size=64, verbose=0,
        )
        probs = preds[0] if isinstance(preds, list) else preds
        y_pred = probs.argmax(axis=1)

        acc = float((y_pred == y).mean())
        cm = confusion_matrix(y, y_pred, labels=list(range(NUM_CLASSES)))
        report = classification_report(
            y, y_pred, labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )

        tomato = pair_confusion(y, y_pred, 3, 4)  # tomato early vs late
        potato = pair_confusion(y, y_pred, 7, 8)  # potato early vs late

        results[name] = {
            "val_accuracy": acc,
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class_f1": {n: report[n]["f1-score"] for n in CLASS_NAMES},
            "early_vs_late_tomato": {
                "correct": tomato[0], "total": tomato[1],
                "accuracy": tomato[2],
            },
            "early_vs_late_potato": {
                "correct": potato[0], "total": potato[1],
                "accuracy": potato[2],
            },
            "confusion_matrix": cm.tolist(),
        }

        print(f"  val acc:            {acc:.4f}")
        print(f"  macro F1:           {report['macro avg']['f1-score']:.4f}")
        print(f"  tomato early/late:  {tomato[2]:.4f}  ({tomato[0]}/{tomato[1]})")
        print(f"  potato early/late:  {potato[2]:.4f}  ({potato[0]}/{potato[1]})")

        # Show early/late confusion detail
        print(f"  confusion slice (tomato E vs L):")
        print(f"    early → early: {cm[3,3]}   early → late: {cm[3,4]}")
        print(f"    late  → early: {cm[4,3]}   late  → late: {cm[4,4]}")
        print(f"  confusion slice (potato E vs L):")
        print(f"    early → early: {cm[7,7]}   early → late: {cm[7,8]}")
        print(f"    late  → early: {cm[8,7]}   late  → late: {cm[8,8]}")

    # Save
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / "per_class_analysis.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved: {out}")

    # ─── Summary table ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print("PER-CLASS SUMMARY")
    print("=" * 90)
    print(f"{'Model':<26} {'Val':>8} {'MacroF1':>9} "
          f"{'TomE/L':>9} {'PotE/L':>9}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<26} {r['val_accuracy']:>8.4f} "
              f"{r['macro_f1']:>9.4f} "
              f"{r['early_vs_late_tomato']['accuracy']:>9.4f} "
              f"{r['early_vs_late_potato']['accuracy']:>9.4f}")

    # ─── H1 focus: attention vs concat on hard pairs ─────────────
    print("\n" + "=" * 90)
    print("H1 FOCUS — attention vs concat on early-vs-late pairs")
    print("=" * 90)
    if "context_plus_4h" in results and "context_concat" in results:
        a = results["context_plus_4h"]
        c = results["context_concat"]
        print(f"  Aggregate:   attn={a['val_accuracy']:.4f}  "
              f"concat={c['val_accuracy']:.4f}  "
              f"Δ={(a['val_accuracy']-c['val_accuracy'])*100:+.2f} pp")
        print(f"  Tomato E/L:  attn={a['early_vs_late_tomato']['accuracy']:.4f}  "
              f"concat={c['early_vs_late_tomato']['accuracy']:.4f}  "
              f"Δ={(a['early_vs_late_tomato']['accuracy'] - c['early_vs_late_tomato']['accuracy'])*100:+.2f} pp")
        print(f"  Potato E/L:  attn={a['early_vs_late_potato']['accuracy']:.4f}  "
              f"concat={c['early_vs_late_potato']['accuracy']:.4f}  "
              f"Δ={(a['early_vs_late_potato']['accuracy'] - c['early_vs_late_potato']['accuracy'])*100:+.2f} pp")
    else:
        print("  ⚠️  Need both context_plus_4h and context_concat results")


if __name__ == "__main__":
    main()