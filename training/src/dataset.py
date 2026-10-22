"""
PlantVillage 10-class loader for AgriYouthNet-Context+.

Loads from a local class-foldered directory (no TFDS dependency).
Context vector = 4-dim one-hot crop type (maize / tomato / potato / other),
derived from the class label — no farmer metadata needed for training.

Returns tf.data.Dataset yielding:
  inputs:  {"leaf_image": (B,224,224,3) f32, "context_vector": (B,4) f32}
  targets: {"class_probs": (B,10) f32 onehot, "attention_weights": (B,4) f32 zeros}
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# ─── Augmentation toggle (for debugging) ─────────────────────────
_AUGMENTATION_ENABLED = True

def set_augmentation_enabled(enabled: bool):
    """Toggle augmentation globally (used by train.py --no-aug)."""
    global _AUGMENTATION_ENABLED
    _AUGMENTATION_ENABLED = enabled
    print(f"  → Augmentation {'ENABLED' if enabled else 'DISABLED'}")


# ─── Class definitions ───────────────────────────────────────────
CLASS_NAMES = [
    "Maize_Northern_Leaf_Blight",   # 0
    "Maize_Common_Rust",            # 1
    "Maize_Healthy",                # 2
    "Tomato_Early_Blight",          # 3
    "Tomato_Late_Blight",           # 4
    "Tomato_Leaf_Mold",             # 5
    "Tomato_Healthy",               # 6
    "Potato_Early_Blight",          # 7
    "Potato_Late_Blight",           # 8
    "Potato_Healthy",               # 9
]

# PlantVillage raw class name → our 0-9 index
# These match the actual folder names in the abdallahalidev/plantvillage-dataset mirror.
PV_TO_INDEX = {
    "Corn_(maize)___Northern_Leaf_Blight": 0,
    "Corn_(maize)___Common_rust_":         1,
    "Corn_(maize)___healthy":              2,
    "Tomato___Early_blight":               3,
    "Tomato___Late_blight":                4,
    "Tomato___Leaf_Mold":                  5,
    "Tomato___healthy":                    6,
    "Potato___Early_blight":               7,
    "Potato___Late_blight":                8,
    "Potato___healthy":                    9,
}

# Crop type per class (for context one-hot)
CROP_PER_CLASS = [
    "maize", "maize", "maize",
    "tomato", "tomato", "tomato", "tomato",
    "potato", "potato", "potato",
]
CROP_INDEX = {"maize": 0, "tomato": 1, "potato": 2, "other": 3}

NUM_CLASSES = 10
CONTEXT_DIM = 4
IMG_SIZE = 224

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─── Augmentation ────────────────────────────────────────────────
def _build_augmenter() -> tf.keras.Sequential:
    """Training-only augmentation. Not saved in the model."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),      # ±15%
        tf.keras.layers.RandomZoom(0.10),          # ±10%
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomBrightness(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ], name="augmenter")


# ─── Preprocessing ───────────────────────────────────────────────
def _preprocess(image: tf.Tensor, training: bool) -> tf.Tensor:
    """Resize → float32 → ImageNet normalize. `image` in [0,255]."""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    image = (image / 255.0 - MEAN) / STD
    return image


# ─── Dataset builders ────────────────────────────────────────────
def _make_tf_dataset(
    paths: List[str],
    labels: List[int],
    training: bool,
    batch_size: int,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build tf.data pipeline from file paths + labels."""
    paths_t = tf.constant(paths)
    labels_t = tf.constant(labels, dtype=tf.int32)
    crops_t = tf.constant([CROP_INDEX[CROP_PER_CLASS[l]] for l in labels],
                          dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths_t, labels_t, crops_t))

    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 2000),
                        seed=seed, reshuffle_each_iteration=True)

    augmenter = _build_augmenter()

    def _load(path, label, crop_idx):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.ensure_shape(img, [None, None, 3])
        img = _preprocess(img, training=training)
        if training:
            img = augmenter(img, training=True)
            img = tf.clip_by_value(img, -3.0, 3.0)
        ctx = tf.one_hot(crop_idx, CONTEXT_DIM, dtype=tf.float32)
        # Attention target is dummy zeros — loss weight on it is 0.
        attn_target = tf.zeros([CONTEXT_DIM], dtype=tf.float32)
        return (
            {"leaf_image": img, "context_vector": ctx},
            {"class_probs": tf.one_hot(label, NUM_CLASSES, dtype=tf.float32),
             "attention_weights": attn_target},
        )

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _scan_local_dir(root: Path) -> Tuple[List[str], List[int]]:
    """Walk a class-foldered directory and return (paths, labels)."""
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    paths, labels = [], []
    for class_name, idx in PV_TO_INDEX.items():
        class_dir = root / class_name
        if not class_dir.exists():
            print(f"  ⚠️  missing class dir: {class_dir}")
            continue
        for p in class_dir.rglob("*"):
            if p.suffix in exts:
                paths.append(str(p))
                labels.append(idx)
    return paths, labels


def _balance_by_class(paths: List[str], labels: List[int],
                      max_per_class: Optional[int], seed: int = 42) -> Tuple[List[str], List[int]]:
    """Cap each class to max_per_class samples (shuffled)."""
    if max_per_class is None:
        return paths, labels
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[str]] = {}
    for p, l in zip(paths, labels):
        by_class.setdefault(l, []).append(p)
    out_p, out_l = [], []
    for c, plist in by_class.items():
        rng.shuffle(plist)
        plist = plist[:max_per_class]
        out_p.extend(plist)
        out_l.extend([c] * len(plist))
    return out_p, out_l


def _stratified_split(paths, labels, val_ratio: float, seed: int = 42):
    """80/20 stratified split."""
    rng = np.random.default_rng(seed)
    by_class: Dict[int, List[int]] = {}
    for i, l in enumerate(labels):
        by_class.setdefault(l, []).append(i)
    train_idx, val_idx = [], []
    for c, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_ratio))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return ([paths[i] for i in train_idx], [labels[i] for i in train_idx],
            [paths[i] for i in val_idx],   [labels[i] for i in val_idx])


# ─── Public API ──────────────────────────────────────────────────
def load_from_local(
    root: str = "training/data/plantvillage",
    val_ratio: float = 0.2,
    batch_size: int = 32,
    max_per_class: Optional[int] = 500,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Load PlantVillage from a local class-foldered directory.

    Expected layout:
      training/data/plantvillage/
        Corn_(maize)___Northern_Leaf_Blight/
        Corn_(maize)___Common_rust_/
        Corn_(maize)___healthy/
        Tomato___Early_blight/
        Tomato___Late_blight/
        Tomato___Leaf_Mold/
        Tomato___healthy/
        Potato___Early_blight/
        Potato___Late_blight/
        Potato___healthy/
    """
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"Local dataset not found: {root_p}")

    print(f"Scanning {root_p} ...")
    paths, labels = _scan_local_dir(root_p)
    if not paths:
        raise RuntimeError(f"No images found under {root_p}")

    paths, labels = _balance_by_class(paths, labels, max_per_class, seed)
    print(f"  Total after balancing (max {max_per_class}/class): {len(paths)}")

    tr_p, tr_l, va_p, va_l = _stratified_split(paths, labels, val_ratio, seed)
    print(f"  Train: {len(tr_p)}   Val: {len(va_p)}")

    train_ds = _make_tf_dataset(tr_p, tr_l, training=True,  batch_size=batch_size, seed=seed)
    val_ds   = _make_tf_dataset(va_p, va_l, training=False, batch_size=batch_size, seed=seed)

    info = {
        "source": "local",
        "root": str(root_p),
        "num_classes": NUM_CLASSES,
        "context_dim": CONTEXT_DIM,
        "train_size": len(tr_p),
        "val_size": len(va_p),
        "class_names": CLASS_NAMES,
    }
    return train_ds, val_ds, info


def load_auto(
    local_root: Optional[str] = None,
    **kwargs,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, Dict]:
    """
    Try several plausible local paths. No TFDS fallback (broken on Python 3.12
    with TF 2.16.1 due to protobuf version conflict).

    Paths checked in order:
      - user-provided local_root
      - <cwd>/data/plantvillage
      - <cwd>/training/data/plantvillage
      - <script_dir>/../data/plantvillage
    """
    candidates = []
    if local_root:
        candidates.append(Path(local_root))
    candidates += [
        Path("data/plantvillage"),
        Path("training/data/plantvillage"),
        Path(__file__).resolve().parent.parent / "data" / "plantvillage",
    ]

    sample_class = "Corn_(maize)___Northern_Leaf_Blight"

    for cand in candidates:
        if cand.exists() and cand.is_dir() and (cand / sample_class).exists():
            print(f"✓ Using local dataset: {cand.resolve()}")
            return load_from_local(str(cand), **kwargs)

    # Nothing worked — print diagnostics
    print("\n❌ No PlantVillage dataset found in any of:")
    for c in candidates:
        status = "exists" if c.exists() else "missing"
        print(f"   - {c.resolve() if c.exists() else c}   [{status}]")
    print("\nExpected one folder per class, e.g.:")
    print("   data/plantvillage/Corn_(maize)___Northern_Leaf_Blight/")
    print("   data/plantvillage/Tomato___Late_blight/")
    print("\nCurrent working directory:")
    print(f"   {Path.cwd()}")
    raise FileNotFoundError(
        "PlantVillage not found. Place it at data/plantvillage/ or "
        "pass local_root=... to load_auto()."
    )


# ─── CLI ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Dataset loader smoke test")
    print("=" * 60)

    try:
        train_ds, val_ds, info = load_auto(batch_size=4, max_per_class=20)
    except Exception as e:
        print(f"\n❌ Could not load dataset: {type(e).__name__}: {e}")
        raise SystemExit(1)

    print("\nDataset info:")
    for k, v in info.items():
        if k != "class_names":
            print(f"  {k}: {v}")

    print("\nBatch inspection:")
    for (inputs, targets) in train_ds.take(1):
        print(f"  leaf_image shape:        {inputs['leaf_image'].shape}")
        print(f"  context_vector shape:    {inputs['context_vector'].shape}")
        print(f"  class_probs shape:       {targets['class_probs'].shape}")
        print(f"  attention_weights shape: {targets['attention_weights'].shape}")

        assert inputs['leaf_image'].shape == (4, 224, 224, 3), "image batch shape wrong"
        assert inputs['context_vector'].shape == (4, 4), "context shape wrong"
        assert targets['class_probs'].shape == (4, 10), "probs shape wrong"
        assert targets['attention_weights'].shape == (4, 4), "attn shape wrong"

        print("\n  Sample class_probs (should be one-hot):")
        print(f"    {targets['class_probs'][0].numpy()}")
        print("  Sample context_vector (should be one-hot):")
        print(f"    {inputs['context_vector'][0].numpy()}")

    print("\n✅ Dataset loader works.")