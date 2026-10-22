"""
Retrain AgriYouthNet-Context+ with field-realistic augmentation.

Fix vs earlier attempt:
  - Augmentation applied in [0, 1] range (matches model input)
  - No -3.0/3.0 clipping
  - Added RandomCrop (simulates framing variance)
  - Added RandomSaturation (simulates lighting color variance)
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPT_DIR))

import json, time
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

DATA_ROOT = _SCRIPT_DIR / "data" / "plantvillage"
OUT_DIR = _SCRIPT_DIR / "models" / "context_plus_4h_aug"
LOG_DIR = _SCRIPT_DIR / "logs" / "context_plus_4h_aug"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
N_PER_CLASS = 500
BATCH_SIZE = 32
EPOCHS = 40
LR = 3e-4
SEED = 42
CONTEXT_DIM = 4
NUM_CLASSES = 10


def load_data():
    print(f"Loading images from {DATA_ROOT}...")
    X_img, X_ctx, y = [], [], []
    for class_name, idx in PV_TO_INDEX.items():
        folder = DATA_ROOT / class_name
        files = (list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))
                 + list(folder.glob("*.PNG")) + list(folder.glob("*.png")))
        files = files[:N_PER_CLASS]
        crop_idx = CROP_INDEX[CROP_PER_CLASS[idx]]
        for f in files:
            img = Image.open(f).convert("RGB")
            # Crop to square BEFORE resize (matches mobile fix)
            w, h = img.size
            s = min(w, h)
            x_off = (w - s) // 2
            y_off = (h - s) // 2
            img = img.crop((x_off, y_off, x_off + s, y_off + s))
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            X_img.append(np.array(img, dtype=np.float32) / 255.0)
            ctx = np.zeros(CONTEXT_DIM, dtype=np.float32); ctx[crop_idx] = 1.0
            X_ctx.append(ctx)
            y.append(idx)
    return np.stack(X_img), np.stack(X_ctx), np.array(y, dtype=np.int32)


def make_augmenter():
    """Augmentation in [0,1] range. Simulates field conditions."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.12, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(0.15, fill_mode="reflect"),
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="reflect"),
        tf.keras.layers.RandomBrightness(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augmenter")


def make_train_ds(X_img, X_ctx, y_onehot, augmenter):
    ds = tf.data.Dataset.from_tensor_slices(
        (X_img.astype(np.float32),
         X_ctx.astype(np.float32),
         y_onehot.astype(np.float32))
    )
    ds = ds.shuffle(len(X_img), seed=SEED, reshuffle_each_iteration=True)

    def _aug(img, ctx, y):
        img = augmenter(img, training=True)
        img = tf.clip_by_value(img, 0.0, 1.0)   # keep in [0,1] — model was trained there
        return (
            {"leaf_image": img, "context_vector": ctx},
            {"class_probs": y, "attention_weights": tf.zeros([CONTEXT_DIM])},
        )

    return ds.map(_aug, num_parallel_calls=tf.data.AUTOTUNE)\
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def make_val_ds(X_img, X_ctx, y_onehot):
    ds = tf.data.Dataset.from_tensor_slices(
        (X_img.astype(np.float32),
         X_ctx.astype(np.float32),
         y_onehot.astype(np.float32))
    )
    def _p(img, ctx, y):
        return (
            {"leaf_image": img, "context_vector": ctx},
            {"class_probs": y, "attention_weights": tf.zeros([CONTEXT_DIM])},
        )
    return ds.map(_p, num_parallel_calls=tf.data.AUTOTUNE)\
             .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    tf.random.set_seed(SEED); np.random.seed(SEED)

    print("=" * 60)
    print("TRAINING: context_plus_4h with augmentation v2 (field-realistic)")
    print("=" * 60)

    X_img, X_ctx, y = load_data()
    print(f"  X_img: {X_img.shape}  range=[{X_img.min():.3f}, {X_img.max():.3f}]")
    y_onehot = np.eye(NUM_CLASSES, dtype=np.float32)[y]

    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y, random_state=SEED)
    print(f"  Train: {len(tr)}  Val: {len(va)}")

    train_ds = make_train_ds(X_img[tr], X_ctx[tr], y_onehot[tr], make_augmenter())
    val_ds = make_val_ds(X_img[va], X_ctx[va], y_onehot[va])

    model = build_agriyouthnet_context_plus(
        num_heads=4, fusion="attention", width="wide", name="context_plus_4h_aug")
    print(f"  Params: {model.count_params():,}")

    model.compile(
        optimizer=Adam(learning_rate=LR, clipnorm=1.0),
        loss={"class_probs": CategoricalCrossentropy(), "attention_weights": "mse"},
        loss_weights={"class_probs": 1.0, "attention_weights": 0.0},
        metrics={"class_probs": [CategoricalAccuracy(name="accuracy")]},
    )

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

    print("\nTraining...")
    t0 = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks, verbose=1,
    )
    print(f"\nTraining time: {(time.time()-t0)/60:.1f} min")

    best = float(max(history.history["val_class_probs_accuracy"]))
    final = model.evaluate(val_ds, verbose=0, return_dict=True)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Params:              {model.count_params():,}")
    print(f"  Best val accuracy:   {best:.4f}")
    print(f"  Final val accuracy:  {final['class_probs_accuracy']:.4f}")
    print(f"  Epochs run:          {len(history.history['loss'])}")

    summary = {
        "model": "context_plus_4h_aug",
        "params": int(model.count_params()),
        "val_accuracy": best,
        "final_val_accuracy": float(final["class_probs_accuracy"]),
        "epochs_run": len(history.history["loss"]),
        "n_per_class": N_PER_CLASS,
        "augmentation": "v2-field-realistic",
        "train_size": int(len(tr)),
        "val_size": int(len(va)),
    }
    (LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    model.save(str(OUT_DIR / "final.keras"))
    print(f"\n✓ Saved to {OUT_DIR}")


if __name__ == "__main__":
    main()