"""
Unified training loop for AgriYouthNet-Context+ and ablations.

Key changes in this version:
  - Uses in-memory best weights (no reload → no serialization bug)
  - Optional --no-aug flag for debugging
  - Diagnostic output on first batch
  - LR default 3e-4 (safer than 5e-4)
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
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)

from src.dataset import (
    load_auto, NUM_CLASSES, CONTEXT_DIM, CLASS_NAMES,
    set_augmentation_enabled,
)
from src.agriyouthnet_context_plus import (
    build_agriyouthnet_context_plus,
    build_agriyouthnet_no_context,
)


# ─── Model registry ──────────────────────────────────────────────
def _wrap_no_context():
    base = build_agriyouthnet_no_context(width="wide")
    img = layers.Input(shape=(224, 224, 3), name="leaf_image")
    ctx = layers.Input(shape=(CONTEXT_DIM,), name="context_vector")
    probs = base(img)
    attn = layers.Lambda(lambda t: t * 0.0, output_shape=(CONTEXT_DIM,),
                         name="attention_weights")(ctx)
    return Model([img, ctx], [probs, attn], name="no_context_wrapped")


MODEL_BUILDERS = {
    "context_plus_4h": lambda: build_agriyouthnet_context_plus(
        num_heads=4, fusion="attention", width="wide", name="context_plus_4h"),
    "context_plus_1h": lambda: build_agriyouthnet_context_plus(
        num_heads=1, fusion="attention", width="wide", name="context_plus_1h"),
    "context_plus_8h": lambda: build_agriyouthnet_context_plus(
        num_heads=8, fusion="attention", width="wide", name="context_plus_8h"),
    "context_plus_4h_narrow": lambda: build_agriyouthnet_context_plus(
        num_heads=4, fusion="attention", width="narrow", name="context_plus_4h_narrow"),
    "context_concat": lambda: build_agriyouthnet_context_plus(
        num_heads=1, fusion="concat", width="wide", name="context_concat"),
    "context_gating": lambda: build_agriyouthnet_context_plus(
        num_heads=1, fusion="gating", width="wide", name="context_gating"),
    "no_context": _wrap_no_context,
}


# ─── Diagnostics ─────────────────────────────────────────────────
def _diag_batch(ds, name: str):
    """Print stats on one batch to catch data pipeline issues."""
    for (inputs, targets) in ds.take(1):
        img = inputs["leaf_image"].numpy()
        ctx = inputs["context_vector"].numpy()
        probs = targets["class_probs"].numpy()
        print(f"\n  [{name}] batch diagnostics:")
        print(f"    image:  shape={img.shape} "
              f"min={img.min():.3f} max={img.max():.3f} "
              f"mean={img.mean():.3f} std={img.std():.3f}")
        print(f"    context: shape={ctx.shape} "
              f"row_sums={ctx.sum(axis=1)[:4]}")
        print(f"    probs:  shape={probs.shape} "
              f"row_sums={probs.sum(axis=1)[:4]}")
        print(f"    probs[0] = {probs[0]}")


# ─── Training ────────────────────────────────────────────────────
def train(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    if args.no_aug:
        set_augmentation_enabled(False)
        print("⚠️  Augmentation DISABLED for this run")

    models_dir = Path(args.save_dir) / args.model
    logs_dir = Path(args.log_dir) / args.model
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"TRAINING: {args.model}")
    print(f"Epochs: {args.epochs}  Batch: {args.batch_size}  "
          f"Max/class: {args.max_per_class}  LR: {args.lr}")
    print(f"Augmentation: {'OFF' if args.no_aug else 'ON'}")
    print(f"Save dir: {models_dir}")
    print("=" * 60)

    # ─── Data ────────────────────────────────────────────────────
    print("\n[1/5] Loading dataset...")
    train_ds, val_ds, ds_info = load_auto(
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    steps_per_epoch = ds_info["train_size"] // args.batch_size
    print(f"  Steps/epoch: {steps_per_epoch}")

    _diag_batch(train_ds, "train")
    _diag_batch(val_ds, "val")

    # ─── Model ───────────────────────────────────────────────────
    print("\n[2/5] Building model...")
    model = MODEL_BUILDERS[args.model]()
    n_params = model.count_params()
    print(f"  Params: {n_params:,}")

    # ─── Compile ─────────────────────────────────────────────────
    print("\n[3/5] Compiling...")
    model.compile(
        optimizer=Adam(learning_rate=args.lr, clipnorm=1.0),
        loss={
            "class_probs": CategoricalCrossentropy(),
            "attention_weights": "mse",
        },
        loss_weights={"class_probs": 1.0, "attention_weights": 0.0},
        metrics={"class_probs": [CategoricalAccuracy(name="accuracy")]},
    )

    # ─── Callbacks ───────────────────────────────────────────────
    ckpt_path = models_dir / "best.keras"
    csv_path = logs_dir / "history.csv"

    callbacks = [
        ModelCheckpoint(str(ckpt_path), monitor="val_class_probs_accuracy",
                        mode="max", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_class_probs_accuracy", mode="max",
                      patience=args.patience, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4,
                          min_lr=1e-6, verbose=1),
        CSVLogger(str(csv_path), append=False),
    ]

    # ─── Train ───────────────────────────────────────────────────
    print("\n[4/5] Training...")
    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"\n  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # ─── Evaluate in-memory model (best weights already restored) ─
    print("\n[5/5] Evaluating (in-memory best weights)...")
    eval_results = model.evaluate(val_ds, verbose=0, return_dict=True)
    val_acc = eval_results.get("class_probs_accuracy", 0)
    print(f"  Val loss:     {eval_results.get('loss', 0):.4f}")
    print(f"  Val accuracy: {val_acc:.4f}")

    # ─── Summary ─────────────────────────────────────────────────
    summary = {
        "model": args.model,
        "params": n_params,
        "epochs_run": len(history.history["loss"]),
        "training_time_s": round(elapsed, 1),
        "best_val_accuracy": float(max(history.history["val_class_probs_accuracy"])),
        "final_val_accuracy": float(val_acc),
        "final_val_loss": float(eval_results.get("loss", 0)),
        "augmentation": "off" if args.no_aug else "on",
        "lr": args.lr,
        "batch_size": args.batch_size,
        "max_per_class": args.max_per_class,
        "seed": args.seed,
        "train_size": ds_info["train_size"],
        "val_size": ds_info["val_size"],
        "class_names": CLASS_NAMES,
    }
    with open(logs_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save final model (in-memory, best weights)
    final_path = models_dir / "final.keras"
    model.save(str(final_path))
    print(f"\n✓ Final model:  {final_path}")
    print(f"✓ Best ckpt:    {ckpt_path}")
    print(f"✓ History CSV:  {csv_path}")
    print(f"✓ Summary:      {logs_dir / 'summary.json'}")

    return summary


# ─── CLI ─────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="context_plus_4h",
                   choices=list(MODEL_BUILDERS.keys()))
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-per-class", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-aug", action="store_true",
                   help="Disable augmentation (for debugging)")
    p.add_argument("--save-dir", default="training/models")
    p.add_argument("--log-dir", default="training/logs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = train(args)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        if k == "class_names":
            continue
        print(f"  {k}: {v}")