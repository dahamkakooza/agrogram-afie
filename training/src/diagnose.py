"""
Standalone diagnostic. Identifies where the training pipeline breaks.
Run: python src/diagnose.py
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

from src.dataset import load_auto, CLASS_NAMES, NUM_CLASSES

print("=" * 60)
print("DIAGNOSTIC — isolating the training pipeline")
print("=" * 60)

# ─── 1. Load a tiny dataset ─────────────────────────────────────
print("\n[1] Loading tiny dataset (10/class)...")
train_ds, val_ds, info = load_auto(batch_size=10, max_per_class=10)

# ─── 2. Inspect ONE batch in detail ─────────────────────────────
print("\n[2] Inspecting one training batch...")
for (inputs, targets) in train_ds.take(1):
    img = inputs["leaf_image"].numpy()
    ctx = inputs["context_vector"].numpy()
    probs = targets["class_probs"].numpy()

    print(f"  image shape:   {img.shape}")
    print(f"  image dtype:   {img.dtype}")
    print(f"  image range:   [{img.min():.3f}, {img.max():.3f}]")
    print(f"  image mean:    {img.mean():.3f}")
    print(f"  image std:     {img.std():.3f}")
    print(f"  context shape: {ctx.shape}")
    print(f"  context sums:  {ctx.sum(axis=1)}")
    print(f"  probs shape:   {probs.shape}")
    print(f"  probs sums:    {probs.sum(axis=1)}")
    print(f"  probs argmax:  {probs.argmax(axis=1)}")

    # Check: are images from different classes actually different?
    print("\n[3] Are different classes visually different?")
    class_means = []
    labels_in_batch = probs.argmax(axis=1)
    for i in range(len(img)):
        class_means.append((labels_in_batch[i], img[i].mean()))
    print(f"  (class, image_mean) pairs: {class_means}")
    uniq_classes = set(labels_in_batch)
    if len(uniq_classes) > 1:
        means_by_class = {}
        for c, m in class_means:
            means_by_class.setdefault(c, []).append(m)
        for c, ms in means_by_class.items():
            print(f"    class {c}: mean={np.mean(ms):.3f}")

    # Check: per-class pixel histograms (should differ)
    print("\n[4] Per-class variance across all pixels:")
    for c in sorted(set(labels_in_batch)):
        class_imgs = img[labels_in_batch == c]
        print(f"    class {c}: n={len(class_imgs)} "
              f"pixel_mean={class_imgs.mean():.3f} "
              f"pixel_std={class_imgs.std():.3f}")

# ─── 5. Simple baseline: can a LINEAR model on downsampled pixels learn? ───
print("\n[5] Logistic regression baseline on 32x32 downsampled pixels...")
print("    (If this fails, the data itself is broken.)")

# Load more data for the baseline
train_ds_full, val_ds_full, _ = load_auto(batch_size=64, max_per_class=100)
X_list, y_list = [], []
for (inp, tgt) in train_ds_full:
    X_list.append(inp["leaf_image"].numpy())
    y_list.append(tgt["class_probs"].numpy().argmax(axis=1))
X = np.concatenate(X_list, axis=0)
y = np.concatenate(y_list, axis=0)
print(f"    X shape: {X.shape}  y shape: {y.shape}")
print(f"    class distribution: {np.bincount(y)}")

# Downsample to 32x32
X_small = tf.image.resize(X, (32, 32)).numpy()
X_flat = X_small.reshape(len(X_small), -1)
X_flat = (X_flat - X_flat.mean()) / (X_flat.std() + 1e-6)

# Train simple logistic regression
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam

lin_model = tf.keras.Sequential([
    layers.Input(shape=(X_flat.shape[1],)),
    layers.Dense(64, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])
lin_model.compile(
    optimizer=Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
history = lin_model.fit(X_flat, y, epochs=20, batch_size=32,
                        validation_split=0.2, verbose=1)
print(f"\n    Linear baseline final train acc: {history.history['accuracy'][-1]:.3f}")
print(f"    Linear baseline final val acc:   {history.history['val_accuracy'][-1]:.3f}")

# ─── 6. Test the actual model forward pass on a single batch ────
print("\n[6] Testing AgriYouthNet-Context+ forward pass...")
from src.agriyouthnet_context_plus import build_agriyouthnet_context_plus

model = build_agriyouthnet_context_plus(num_heads=4, fusion="attention", width="wide")
print(f"    params: {model.count_params():,}")

for (inp, tgt) in train_ds.take(1):
    out = model(inp, training=False)
    probs, attn = out
    print(f"    forward probs shape: {probs.shape}")
    print(f"    forward probs row sums: {probs.numpy().sum(axis=1)}")
    print(f"    forward attn row sums: {attn.numpy().sum(axis=1)}")

# ─── 7. Overfit a single batch ──────────────────────────────────
print("\n[7] Can the model overfit ONE batch? (Should hit ~100% train acc)")
for (inp, tgt) in train_ds.take(1):
    single_inputs = inp
    single_targets = tgt

model.compile(
    optimizer=Adam(1e-3),
    loss={"class_probs": "categorical_crossentropy", "attention_weights": "mse"},
    loss_weights={"class_probs": 1.0, "attention_weights": 0.0},
    metrics={"class_probs": ["accuracy"]},
)

history = model.fit(single_inputs, single_targets, epochs=30, verbose=0)
final_acc = history.history["class_probs_accuracy"][-1]
print(f"    Single-batch final train accuracy: {final_acc:.3f}")
if final_acc > 0.9:
    print("    ✅ Model CAN learn — data pipeline is the problem")
else:
    print("    ❌ Model cannot overfit one batch — architecture problem")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)