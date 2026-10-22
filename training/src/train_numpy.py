"""
Training — PIL load, no augmentation. Proven 96.4% at N=400.
Bumping N to 500 should push to 97-98%.
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger,
)

from src.agriyouthnet_context_plus import build_agriyouthnet_context_plus
from src.dataset import PV_TO_INDEX, CROP_PER_CLASS, CROP_INDEX

# ─── Config ──────────────────────────────────────────────────────
ROOT = Path("data/plantvillage")
IMG_SIZE = 224
N_PER_CLASS = 500
BATCH_SIZE = 64
EPOCHS = 35
LR = 3e-4
SEED = 42
CONTEXT_DIM = 4
NUM_CLASSES = 10

OUT_DIR = Path("training/models/context_plus_4h_noaug")
LOG_DIR = Path("training/logs/context_plus_4h_noaug")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load ────────────────────────────────────────────────────────
print(f"Loading images via PIL (max {N_PER_CLASS}/class)...")
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
y_onehot = np.eye(NUM_CLASSES, dtype=np.float32)[y]
print(f"  X_img: {X_img.shape}  y dist: {np.bincount(y, minlength=NUM_CLASSES)}")

# ─── Split ───────────────────────────────────────────────────────
idx = np.arange(len(y))
tr, va = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
print(f"  Train: {len(tr)}   Val: {len(va)}")

# ─── Model ───────────────────────────────────────────────────────
print("\nBuilding model...")
model = build_agriyouthnet_context_plus(num_heads=4, fusion="attention", width="wide")
print(f"  params: {model.count_params():,}")

model.compile(
    optimizer=Adam(learning_rate=LR, clipnorm=1.0),
    loss={"class_probs": CategoricalCrossentropy(), "attention_weights": "mse"},
    loss_weights={"class_probs": 1.0, "attention_weights": 0.0},
    metrics={"class_probs": [CategoricalAccuracy(name="accuracy")]},
)

# ─── Callbacks ───────────────────────────────────────────────────
ckpt = OUT_DIR / "best.keras"
callbacks = [
    ModelCheckpoint(str(ckpt), monitor="val_class_probs_accuracy",
                    mode="max", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_class_probs_accuracy", mode="max",
                  patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                      min_lr=1e-6, verbose=1),
    CSVLogger(str(LOG_DIR / "history.csv"), append=False),
]

# ─── Train ───────────────────────────────────────────────────────
print("\nTraining...")
t0 = time.time()
history = model.fit(
    {"leaf_image": X_img[tr], "context_vector": X_ctx[tr]},
    {"class_probs": y_onehot[tr],
     "attention_weights": np.zeros((len(tr), CONTEXT_DIM), dtype=np.float32)},
    validation_data=(
        {"leaf_image": X_img[va], "context_vector": X_ctx[va]},
        {"class_probs": y_onehot[va],
         "attention_weights": np.zeros((len(va), CONTEXT_DIM), dtype=np.float32)},
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1,
)
print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")

# ─── Summary ─────────────────────────────────────────────────────
best_acc = float(max(history.history["val_class_probs_accuracy"]))
final = model.evaluate(
    {"leaf_image": X_img[va], "context_vector": X_ctx[va]},
    {"class_probs": y_onehot[va],
     "attention_weights": np.zeros((len(va), CONTEXT_DIM), dtype=np.float32)},
    verbose=0, return_dict=True,
)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Params:             {model.count_params():,}")
print(f"  Best val accuracy:  {best_acc:.4f}")
print(f"  Final val accuracy: {final['class_probs_accuracy']:.4f}")
print(f"  Final val loss:     {final['loss']:.4f}")
print(f"  Epochs run:         {len(history.history['loss'])}")

summary = {
    "model": "context_plus_4h",
    "params": int(model.count_params()),
    "val_accuracy": best_acc,
    "final_val_accuracy": float(final["class_probs_accuracy"]),
    "final_val_loss": float(final["loss"]),
    "epochs_run": len(history.history["loss"]),
    "n_per_class": N_PER_CLASS,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "augmentation": "off",
    "train_size": int(len(tr)),
    "val_size": int(len(va)),
}
with open(LOG_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

model.save(str(OUT_DIR / "final.keras"))
print(f"\n✓ Best:  {ckpt}")
print(f"✓ Final: {OUT_DIR / 'final.keras'}")