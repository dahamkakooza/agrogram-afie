"""
Minimal data integrity check — bypasses tf.data entirely.
Uses PIL + sklearn to determine if the raw dataset has class-separating signal.
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

ROOT = Path("data/plantvillage")

# Pick a few classes with distinctive visual appearance
TARGETS = [
    ("Corn_(maize)___Northern_Leaf_Blight", 0),
    ("Corn_(maize)___Common_rust_",         1),
    ("Corn_(maize)___healthy",              2),
    ("Tomato___Late_blight",                4),
    ("Potato___healthy",                    9),
]

print("=" * 60)
print("RAW DATA INTEGRITY CHECK (bypasses tf.data)")
print("=" * 60)

X, y = [], []
for name, idx in TARGETS:
    folder = ROOT / name
    if not folder.exists():
        print(f"❌ Missing: {folder}")
        continue
    files = sorted(list(folder.glob("*.JPG")) + list(folder.glob("*.jpg"))
                   + list(folder.glob("*.PNG")) + list(folder.glob("*.png")))
    print(f"\n{name} (idx={idx}): {len(files)} files")
    for f in files[:15]:
        try:
            img = Image.open(f).convert("RGB").resize((32, 32))
            arr = np.array(img, dtype=np.float32).flatten() / 255.0
            X.append(arr)
            y.append(idx)
        except Exception as e:
            print(f"  ❌ {f.name}: {e}")

X = np.array(X)
y = np.array(y)
print(f"\nTotal: {len(X)} samples, {X.shape[1]} features")
print(f"Label distribution: {np.bincount(y)}")
print(f"Pixel stats: min={X.min():.3f} max={X.max():.3f} "
      f"mean={X.mean():.3f} std={X.std():.3f}")

# ─── Test 1: within-class vs across-class distances ────────────
print("\n[Test 1] Within-class vs across-class L2 distance")
for c in np.unique(y):
    idxs = np.where(y == c)[0]
    within = []
    for i in idxs:
        for j in idxs:
            if i != j:
                within.append(np.linalg.norm(X[i] - X[j]))
    across = []
    for i in idxs:
        for j in np.where(y != c)[0]:
            across.append(np.linalg.norm(X[i] - X[j]))
    ratio = np.mean(within) / np.mean(across) if np.mean(across) > 0 else -1
    marker = "✅" if ratio < 0.9 else "❌"
    print(f"  {marker} class {c}: within={np.mean(within):.2f} "
          f"across={np.mean(across):.2f} ratio={ratio:.3f}")

# ─── Test 2: KNN cross-validation ──────────────────────────────
print("\n[Test 2] KNN 3-fold CV accuracy (chance = 0.20 for 5 classes)")
knn = KNeighborsClassifier(n_neighbors=3)
try:
    scores = cross_val_score(knn, X, y, cv=3, scoring="accuracy")
    print(f"  KNN CV scores: {scores}")
    print(f"  Mean CV accuracy: {scores.mean():.3f}")
    if scores.mean() > 0.7:
        print("  ✅ Data has signal — pipeline is broken")
    elif scores.mean() > 0.4:
        print("  ⚠️  Weak signal — partial pipeline issue")
    else:
        print("  ❌ Data has no signal — labels or images are wrong")
except Exception as e:
    print(f"  KNN failed: {e}")

# ─── Test 3: Same-image sanity ─────────────────────────────────
print("\n[Test 3] Same-image distance should be 0")
if len(X) >= 2:
    d = np.linalg.norm(X[0] - X[0])
    print(f"  distance(X[0], X[0]) = {d:.6f}")

# ─── Test 4: Are all images identical? ─────────────────────────
print("\n[Test 4] Are all loaded images distinct?")
from itertools import combinations
n_pairs = 0
identical_pairs = 0
for i, j in list(combinations(range(len(X)), 2))[:200]:
    if np.allclose(X[i], X[j], atol=1e-6):
        identical_pairs += 1
    n_pairs += 1
print(f"  {identical_pairs}/{n_pairs} pairs are pixel-identical")

# ─── Test 5: Per-class image means ─────────────────────────────
print("\n[Test 5] Per-class image means (should differ)")
for c in np.unique(y):
    idxs = np.where(y == c)[0]
    print(f"  class {c}: mean pixel = {X[idxs].mean():.4f}")

print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)
print("If KNN CV > 0.7  → data OK, pipeline is the bug (fixable)")
print("If KNN CV < 0.4  → data is mislabeled or corrupted (verify with PIL open)")