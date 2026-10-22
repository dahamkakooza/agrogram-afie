"""
Train all ablation variants back-to-back (with path robustness).

Runs:
  1. context_concat         (concat fusion baseline)
  2. context_gating         (gating fusion ablation)
  3. context_plus_1h        (1 attention head)
  4. context_plus_8h        (8 attention heads)
  5. context_plus_4h_narrow (52K param size ablation)

Does NOT retrain context_plus_4h — pulls its result from the prior run's
summary.json (from train_numpy.py).

Outputs:
  training/models/<name>_noaug/best.keras
  training/logs/<name>_noaug/summary.json
  training/logs/ablation_results.json
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
from pathlib import Path

# ─── Path bootstrap ──────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent.parent     # D:\agrogram-afie\training
_PROJECT_ROOT = _SCRIPT_DIR.parent                       # D:\agrogram-afie
sys.path.insert(0, str(_SCRIPT_DIR))                     # allow "from src.xxx"
# ─────────────────────────────────────────────────────────────────

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger,
)

from src.agriyouthnet_context_plus import build_agriyouthnet_context_plus
from src.dataset import PV_TO_INDEX, CROP_PER_CLASS, CROP_INDEX, CLASS_NAMES


# ─── Locate the dataset directory ────────────────────────────────
def _find_data_root() -> Path:
    """Check both possible locations for the PlantVillage folder."""
    candidates = [
        _SCRIPT_DIR / "data" / "plantvillage",
        _SCRIPT_DIR / "training" / "data" / "plantvillage",
    ]
    for c in candidates:
        if c.exists() and (c / "Corn_(maize)___Northern_Leaf_Blight").exists():
            return c
    raise FileNotFoundError(
        "Could not find PlantVillage at:\n  " +
        "\n  ".join(str(c) for c in candidates)
    )


# ─── Config ──────────────────────────────────────────────────────
DATA_ROOT = _find_data_root()
IMG_SIZE = 224
N_PER_CLASS = 500
BATCH_SIZE = 64
EPOCHS = 35
LR = 3e-4
SEED = 42
CONTEXT_DIM = 4
NUM_CLASSES = 10

MODELS_DIR = _SCRIPT_DIR / "models"
LOGS_DIR = _SCRIPT_DIR / "logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Ablation registry ───────────────────────────────────────────
ABLATIONS = [
    {
        "name": "context_concat",
        "builder": lambda: build_agriyouthnet_context_plus(
            num_heads=1, fusion="concat", width="wide",
            name="context_concat"),
        "description": "Concat fusion baseline",
    },
    {
        "name": "context_gating",
        "builder": lambda: build_agriyouthnet_context_plus(
            num_heads=1, fusion="gating", width="wide",
            name="context_gating"),
        "description": "Gating fusion ablation",
    },
    {
        "name": "context_plus_1h",
        "builder": lambda: build_agriyouthnet_context_plus(
            num_heads=1, fusion="attention", width="wide",
            name="context_plus_1h"),
        "description": "1 attention head",
    },
    {
        "name": "context_plus_8h",
        "builder": lambda: build_agriyouthnet_context_plus(
            num_heads=8, fusion="attention", width="wide",
            name="context_plus_8h"),
        "description": "8 attention heads",
    },
    {
        "name": "context_plus_4h_narrow",
        "builder": lambda: build_agriyouthnet_context_plus(
            num_heads=4, fusion="attention", width="narrow",
            name="context_plus_4h_narrow"),
        "description": "4 heads, narrow backbone (~52K params)",
    },
]


# ─── Load dataset once ───────────────────────────────────────────
def load_data():
    print(f"Loading images via PIL (max {N_PER_CLASS}/class) from {DATA_ROOT}...")
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
            arr = np.array(img, dtype=np.float32) / 255.0
            X_img.append(arr)
            ctx = np.zeros(CONTEXT_DIM, dtype=np.float32); ctx[crop_idx] = 1.0
            X_ctx.append(ctx)
            y.append(idx)

    X_img = np.stack(X_img)
    X_ctx = np.stack(X_ctx)
    y = np.array(y, dtype=np.int32)
    y_onehot = np.eye(NUM_CLASSES, dtype=np.float32)[y]
    print(f"  X_img: {X_img.shape}")
    print(f"  y dist: {np.bincount(y, minlength=NUM_CLASSES)}")
    return X_img, X_ctx, y, y_onehot


# ─── Train one variant ───────────────────────────────────────────
def train_variant(spec, X_img_tr, X_ctx_tr, y_tr,
                  X_img_va, X_ctx_va, y_va):
    name = spec["name"]
    print("\n" + "=" * 60)
    print(f"TRAINING: {name}  ({spec['description']})")
    print("=" * 60)

    out_dir = MODELS_DIR / f"{name}_noaug"
    log_dir = LOGS_DIR / f"{name}_noaug"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ─── Model ───────────────────────────────────────────────────
    model = spec["builder"]()
    n_params = model.count_params()
    print(f"  params: {n_params:,}")

    model.compile(
        optimizer=Adam(learning_rate=LR, clipnorm=1.0),
        loss={"class_probs": CategoricalCrossentropy(),
              "attention_weights": "mse"},
        loss_weights={"class_probs": 1.0, "attention_weights": 0.0},
        metrics={"class_probs": [CategoricalAccuracy(name="accuracy")]},
    )

    ckpt = out_dir / "best.keras"
    callbacks = [
        ModelCheckpoint(str(ckpt), monitor="val_class_probs_accuracy",
                        mode="max", save_best_only=True, verbose=0),
        EarlyStopping(monitor="val_class_probs_accuracy", mode="max",
                      patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                          min_lr=1e-6, verbose=1),
        CSVLogger(str(log_dir / "history.csv"), append=False),
    ]

    # ─── Train ───────────────────────────────────────────────────
    t0 = time.time()
    history = model.fit(
        {"leaf_image": X_img_tr, "context_vector": X_ctx_tr},
        {"class_probs": y_tr,
         "attention_weights": np.zeros((len(y_tr), CONTEXT_DIM), dtype=np.float32)},
        validation_data=(
            {"leaf_image": X_img_va, "context_vector": X_ctx_va},
            {"class_probs": y_va,
             "attention_weights": np.zeros((len(y_va), CONTEXT_DIM), dtype=np.float32)},
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0

    best_acc = float(max(history.history["val_class_probs_accuracy"]))

    summary = {
        "name": name,
        "description": spec["description"],
        "params": int(n_params),
        "val_accuracy": best_acc,
        "train_accuracy": float(history.history["class_probs_accuracy"][-1]),
        "epochs_run": len(history.history["loss"]),
        "training_time_s": round(elapsed, 1),
        "n_per_class": N_PER_CLASS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "augmentation": "off",
        "train_size": int(len(y_tr)),
        "val_size": int(len(y_va)),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    model.save(str(out_dir / "final.keras"))

    print(f"\n  {name}: val_acc={best_acc:.4f}  "
          f"params={n_params:,}  time={elapsed/60:.1f}min")
    print(f"  ✓ Best: {ckpt}")
    print(f"  ✓ Summary: {log_dir / 'summary.json'}")

    return summary


# ─── Load prior context_plus_4h result ───────────────────────────
def load_prior_primary():
    """Find and load context_plus_4h's summary from the earlier train_numpy.py run."""
    candidates = [
        LOGS_DIR / "context_plus_4h_noaug" / "summary.json",
        _SCRIPT_DIR / "training" / "logs" / "context_plus_4h_noaug" / "summary.json",
        _SCRIPT_DIR / "logs" / "context_plus_4h_noaug" / "summary.json",
        _PROJECT_ROOT / "training" / "logs" / "context_plus_4h_noaug" / "summary.json",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            print(f"✓ Loaded prior primary result from: {path}")
            print(f"  val_accuracy = {data['val_accuracy']:.4f}")
            return {
                "name": "context_plus_4h",
                "description": "Primary: MHCA, 4 heads, wide",
                "params": data["params"],
                "val_accuracy": data["val_accuracy"],
                "train_accuracy": data.get("train_accuracy"),
                "training_time_s": data.get("training_time_s", 0),
                "prior": True,
            }
    print(f"\n⚠️  No prior context_plus_4h summary found in any of:")
    for c in candidates:
        print(f"   {c}")
    return None


# ─── Main ────────────────────────────────────────────────────────
def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("=" * 60)
    print("ABLATION TRAINING")
    print("=" * 60)
    print(f"Project root: {_PROJECT_ROOT}")
    print(f"Data root:    {DATA_ROOT}")
    print(f"Models dir:   {MODELS_DIR}")
    print(f"Logs dir:     {LOGS_DIR}")
    print(f"Variants:     {len(ABLATIONS)}")
    print(f"Per variant:  {EPOCHS} epochs, batch {BATCH_SIZE}, "
          f"{N_PER_CLASS}/class")
    print("=" * 60)

    # ─── Load prior primary result ───────────────────────────────
    primary = load_prior_primary()

    # ─── Load dataset once ───────────────────────────────────────
    X_img, X_ctx, y, y_onehot = load_data()
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    print(f"  Train: {len(tr)}   Val: {len(va)}")

    X_img_tr, X_ctx_tr = X_img[tr], X_ctx[tr]
    X_img_va, X_ctx_va = X_img[va], X_ctx[va]
    y_tr, y_va = y_onehot[tr], y_onehot[va]

    # ─── Run all ablations ───────────────────────────────────────
    all_summaries = []
    if primary:
        all_summaries.append(primary)

    overall_t0 = time.time()
    for i, spec in enumerate(ABLATIONS, 1):
        print(f"\n\n{'#'*60}")
        print(f"# ABLATION {i}/{len(ABLATIONS)}: {spec['name']}")
        print(f"{'#'*60}")
        try:
            s = train_variant(spec,
                              X_img_tr, X_ctx_tr, y_tr,
                              X_img_va, X_ctx_va, y_va)
            all_summaries.append(s)
        except Exception as e:
            print(f"\n❌ FAILED: {spec['name']}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries.append({"name": spec["name"], "error": str(e)})

    total_time = (time.time() - overall_t0) / 60

    # ─── Results table ───────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"{'Variant':<28} {'Params':>10} {'Val Acc':>10} {'Time':>9}")
    print("-" * 70)
    for s in all_summaries:
        if "error" in s:
            print(f"{s['name']:<28} {'FAILED':>10}")
            continue
        t = s.get("training_time_s", 0)
        t_str = f"{t/60:.1f}min" if t else "prior"
        print(f"{s['name']:<28} {s['params']:>10,} "
              f"{s['val_accuracy']:>10.4f} {t_str:>9}")
    print("-" * 70)
    print(f"Total new training time: {total_time:.1f} min")

    results_path = LOGS_DIR / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\n✓ Combined results: {results_path}")

    # ─── H1 verdict ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("H1 VERDICT  (attention 4-head vs concat)")
    print("=" * 70)
    accs = {s["name"]: s["val_accuracy"]
            for s in all_summaries if "error" not in s}

    if "context_plus_4h" in accs and "context_concat" in accs:
        attn_acc = accs["context_plus_4h"]
        concat_acc = accs["context_concat"]
        delta_pp = (attn_acc - concat_acc) * 100
        print(f"  Attention (4h): {attn_acc:.4f}")
        print(f"  Concat:         {concat_acc:.4f}")
        print(f"  Delta:          {delta_pp:+.2f} pp")
        print(f"  H1 target:      ≥ +1.00 pp")
        print(f"  H1 result:      "
              f"{'✅ SUPPORTED' if delta_pp >= 1.0 else '❌ NOT SUPPORTED'}")
    else:
        missing = [n for n in ("context_plus_4h", "context_concat")
                   if n not in accs]
        print(f"  ⚠️  Cannot compute H1 — missing: {missing}")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()