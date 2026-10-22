"""
Training with clean augmentation pipeline.

Key design:
  - Load with PIL (proven correct)
  - Augment in [0, 1] range (before any normalization)
  - tf.data used ONLY for batching + augmentation, not for I/O
  - ImageNet normalization applied as the last step before the model
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
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
CONTEXT_DIM = 4
NUM_CLASSES = 10

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Augmentation (applied in [0,1] range, before normalization) ─
def make_augmenter():
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.10, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
        tf.keras.layers.RandomBrightness(0.08),
        tf.keras.layers.RandomContrast(0.08),
    ], name="augmenter")


def normalize_batch(img_01):
    """img_01: (B, H, W, 3) in [0,1] → ImageNet normalized."""
    return (img_01 - MEAN) / STD


# ─── Load with PIL ───────────────────────────────────────────────
def load_data(n_per_class):
    print(f"Loading images via PIL (max {n_per_class}/class)...")
    X_img, X_ctx, y = [], [], []
    for class_name, idx in PV_TO_INDEX.items():
        folder = ROOT / class_name
        files = (list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))
                 + list(folder.glob("*.PNG")) + list(folder.glob("*.png")))
        files = files[:n_per_class]
        crop_idx = CROP_INDEX[CROP_PER_CLASS[idx]]
        for f in files:
            img = Image.open(f).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img, dtype=np.float32) / 255.0
            X_img.append(arr)
            ctx = np.zeros(CONTEXT_DIM, dtype=np.float32)
            ctx[crop_idx] = 1.0
            X_ctx.append(ctx)
            y.append(idx)
    return (np.stack(X_img), np.stack(X_ctx),
            np.array(y, dtype=np.int32))


# ─── tf.data wrapper (batching + augmentation only) ──────────────
def make_train_ds(X_img, X_ctx, y_onehot, batch_size, augmenter, seed=42):
    """X_img in [0,1]. Augment, then normalize."""
    ds = tf.data.Dataset.from_tensor_slices(
        (X_img.astype(np.float32),
         X_ctx.astype(np.float32),
         y_onehot.astype(np.float32))
    )
    ds = ds.shuffle(buffer_size=len(X_img), seed=seed, reshuffle_each_iteration=True)

    def _aug(img, ctx, y):
        img = augmenter(img, training=True)
        img = tf.clip_by_value(img, 0.0, 1.0)
        img = normalize_batch(img)
        return (
            {"leaf_image": img, "context_vector": ctx},
            {"class_probs": y, "attention_weights": tf.zeros([CONTEXT_DIM])},
        )

    ds = ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_val_ds(X_img, X_ctx, y_onehot, batch_size):
    """No augmentation, only normalization."""
    X_norm = normalize_batch(X_img.astype(np.float32))
    ds = tf.data.Dataset.from_tensor_slices(
        (X_norm, X_ctx.astype(np.float32), y_onehot.astype(np.float32))
    )
    def _p(img, ctx, y):
        return (
            {"leaf_image": img, "context_vector": ctx},
            {"class_probs": y, "attention_weights": tf.zeros([CONTEXT_DIM])},
        )
    ds = ds.map(_p, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ─── Main ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="context_plus_4h")
    ap.add_argument("--n-per-class", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path("training/models") / f"{args.model}_aug"
    log_dir = Path("training/logs") / f"{args.model}_aug"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TRAINING (with augmentation): {args.model}")
    print(f"N/class: {args.n_per_class}  Batch: {args.batch_size}  "
          f"Epochs: {args.epochs}  LR: {args.lr}")
    print("=" * 60)

    # ─── Data ────────────────────────────────────────────────────
    X_img, X_ctx, y = load_data(args.n_per_class)
    y_onehot = np.eye(NUM_CLASSES, dtype=np.float32)[y]
    print(f"  X_img: {X_img.shape}  "
          f"range=[{X_img.min():.3f}, {X_img.max():.3f}]")
    print(f"  y dist: {np.bincount(y, minlength=NUM_CLASSES)}")

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y,
                              random_state=args.seed)
    print(f"  Train: {len(tr)}   Val: {len(va)}")

    augmenter = make_augmenter()
    train_ds = make_train_ds(X_img[tr], X_ctx[tr], y_onehot[tr],
                             args.batch_size, augmenter, args.seed)
    val_ds = make_val_ds(X_img[va], X_ctx[va], y_onehot[va],
                         args.batch_size)

    # ─── Model ───────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_agriyouthnet_context_plus(
        num_heads=4, fusion="attention", width="wide",
        name=args.model,
    )
    print(f"  params: {model.count_params():,}")

    model.compile(
        optimizer=Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={"class_probs": CategoricalCrossentropy(),
              "attention_weights": "mse"},
        loss_weights={"class_probs": 1.0, "attention_weights": 0.0},
        metrics={"class_probs": [CategoricalAccuracy(name="accuracy")]},
    )

    ckpt = out_dir / "best.keras"
    callbacks = [
        ModelCheckpoint(str(ckpt), monitor="val_class_probs_accuracy",
                        mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_class_probs_accuracy", mode="max",
                      patience=args.patience, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        CSVLogger(str(log_dir / "history.csv"), append=False),
    ]

    print("\nTraining...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")

    # ─── Final summary ───────────────────────────────────────────
    best_acc = float(max(history.history["val_class_probs_accuracy"]))
    final_eval = model.evaluate(val_ds, verbose=0, return_dict=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Params:               {model.count_params():,}")
    print(f"  Best val accuracy:    {best_acc:.4f}")
    print(f"  Final val accuracy:   {final_eval['class_probs_accuracy']:.4f}")
    print(f"  Final val loss:       {final_eval['loss']:.4f}")
    print(f"  Epochs run:           {len(history.history['loss'])}")

    summary = {
        "model": args.model,
        "params": int(model.count_params()),
        "val_accuracy": best_acc,
        "final_val_accuracy": float(final_eval["class_probs_accuracy"]),
        "final_val_loss": float(final_eval["loss"]),
        "epochs_run": len(history.history["loss"]),
        "n_per_class": args.n_per_class,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "augmentation": "on",
        "train_size": int(len(tr)),
        "val_size": int(len(va)),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    final_path = out_dir / "final.keras"
    model.save(str(final_path))
    print(f"\n✓ Best:  {ckpt}")
    print(f"✓ Final: {final_path}")
    print(f"✓ Logs:  {log_dir}")


if __name__ == "__main__":
    main()