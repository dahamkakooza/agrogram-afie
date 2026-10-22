"""
Sanity test: TFLite 2-input + 2-output + INT8 conversion.

Verifies that AgriYouthNet-Context+'s architecture (image + context inputs,
class probs + attention weights outputs) can be:
  1. Built in Keras
  2. Converted to TFLite F32
  3. Quantized to TFLite INT8
  4. Run on-device with parity to Keras

Uses TF_USE_LEGACY_KERAS=1 to avoid the Keras 3 TFLite converter bug in TF 2.16.1.
"""
# ─── MUST be set BEFORE importing tensorflow ─────────────────────
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# ─────────────────────────────────────────────────────────────────

import numpy as np
import tensorflow as tf
from pathlib import Path

print(f"TensorFlow version: {tf.__version__}")

# ─── Config ──────────────────────────────────────────────────────
IMG_SIZE = 224
NUM_CLASSES = 10
CONTEXT_DIM = 4          # crop type one-hot
NUM_HEADS = 2
KEY_DIM = 8
FEAT_DIM = 16
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─── 1. Build minimal multi-io model ─────────────────────────────
def build_model():
    img_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="leaf_image")
    ctx_input = tf.keras.Input(shape=(CONTEXT_DIM,), name="context_vector")

    # Tiny visual backbone (stand-in for AgriYouthNet backbone)
    x = tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")(img_input)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    visual_feat = tf.keras.layers.Dense(FEAT_DIM, name="visual_features")(x)  # (B, 16)

    # Context as a sequence of 4 tokens (one per feature)
    ctx_seq = tf.keras.layers.Dense(
        CONTEXT_DIM * FEAT_DIM, activation="relu"
    )(ctx_input)
    ctx_seq = tf.keras.layers.Reshape((CONTEXT_DIM, FEAT_DIM))(ctx_seq)       # (B, 4, 16)

    # Visual as a single query token
    query = tf.keras.layers.Reshape((1, FEAT_DIM))(visual_feat)               # (B, 1, 16)

    # Multi-head cross-attention: visual queries attend to context keys/values
    attn_out = tf.keras.layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=KEY_DIM, name="mhca"
    )(query, ctx_seq, ctx_seq)                                                # (B, 1, 16)
    attn_out = tf.keras.layers.Reshape((FEAT_DIM,))(attn_out)                 # (B, 16)

    # Interpretable attention weights over the 4 context dims
    attn_weights = tf.keras.layers.Dense(
        CONTEXT_DIM, activation="softmax", name="attention_weights"
    )(
        tf.keras.layers.Concatenate()([visual_feat, ctx_input])
    )                                                                         # (B, 4)

    # Classifier head
    fused = tf.keras.layers.Concatenate()([visual_feat, attn_out])            # (B, 32)
    probs = tf.keras.layers.Dense(
        NUM_CLASSES, activation="softmax", name="class_probs"
    )(fused)                                                                  # (B, 10)

    return tf.keras.Model(
        inputs=[img_input, ctx_input],
        outputs=[probs, attn_weights],
        name="sanity_multi_io",
    )


print("\n[0/3] Building model...")
model = build_model()
model.summary()

# ─── 2. Save Keras ───────────────────────────────────────────────
Path("training/tests/artifacts").mkdir(parents=True, exist_ok=True)
keras_path = "training/tests/artifacts/test_model.keras"
model.save(keras_path)
print(f"\n✓ Keras model saved: {keras_path}")


# ─── 3. Convert to TFLite F32 ────────────────────────────────────
print("\n[1/3] Converting to TFLite F32...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_f32 = converter.convert()
f32_path = "training/tests/artifacts/test_model_f32.tflite"
with open(f32_path, "wb") as f:
    f.write(tflite_f32)
print(f"✓ TFLite F32: {len(tflite_f32) / 1024:.2f} KB")


# ─── 4. Convert to TFLite INT8 ───────────────────────────────────
print("\n[2/3] Converting to TFLite INT8...")

def representative_dataset():
    for _ in range(50):
        img = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
        idx = np.random.randint(0, CONTEXT_DIM)
        ctx = np.eye(CONTEXT_DIM, dtype=np.float32)[idx][None, :]
        yield [img, ctx]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32   # keep float I/O for Android ease
converter.inference_output_type = tf.float32

INT8_OK = False
tflite_int8 = None
try:
    tflite_int8 = converter.convert()
    int8_path = "training/tests/artifacts/test_model_int8.tflite"
    with open(int8_path, "wb") as f:
        f.write(tflite_int8)
    print(f"✓ TFLite INT8: {len(tflite_int8) / 1024:.2f} KB")
    print(f"  Compression: {len(tflite_f32) / len(tflite_int8):.1f}x")
    INT8_OK = True
except Exception as e:
    print(f"❌ INT8 conversion FAILED: {type(e).__name__}: {e}")


# ─── 5. Inference parity check ───────────────────────────────────
print("\n[3/3] Verifying inference parity...")


def run_tflite(tflite_bytes, img, ctx):
    """Run TFLite inference, return dict with probs and attention."""
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inputs = interp.get_input_details()
    outputs = interp.get_output_details()

    # Match by shape, not index — TFLite may reorder
    for inp in inputs:
        shape = inp["shape"]
        if shape[-1] == 3:
            interp.set_tensor(inp["index"], img)
        elif shape[-1] == CONTEXT_DIM:
            interp.set_tensor(inp["index"], ctx)
        else:
            raise RuntimeError(f"Unknown input shape: {shape}")

    interp.invoke()

    results = {}
    for out in outputs:
        val = interp.get_tensor(out["index"])
        if val.shape[-1] == NUM_CLASSES:
            results["probs"] = val
        elif val.shape[-1] == CONTEXT_DIM:
            results["attention"] = val
    return results


test_img = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
test_ctx = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

keras_out = model.predict([test_img, test_ctx], verbose=0)

f32_out = run_tflite(tflite_f32, test_img, test_ctx)
print(f"TFLite F32 — probs: {f32_out['probs'].shape}, attention: {f32_out['attention'].shape}")
print(f"  Probs match Keras:     {np.allclose(keras_out[0], f32_out['probs'], atol=1e-3)}")
print(f"  Attention match Keras: {np.allclose(keras_out[1], f32_out['attention'], atol=1e-3)}")

if INT8_OK:
    int8_out = run_tflite(tflite_int8, test_img, test_ctx)
    print(f"\nTFLite INT8 — probs: {int8_out['probs'].shape}, attention: {int8_out['attention'].shape}")
    print(f"  Probs match Keras (atol=0.05):     {np.allclose(keras_out[0], int8_out['probs'], atol=0.05)}")
    print(f"  Attention match Keras (atol=0.05): {np.allclose(keras_out[1], int8_out['attention'], atol=0.05)}")


# ─── 6. Summary ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SANITY TEST SUMMARY")
print("=" * 60)
print(f"Multi-input export:   ✓ (2 inputs)")
print(f"Multi-output export:  ✓ (2 outputs)")
print(f"F32 TFLite export:    ✓ ({len(tflite_f32)/1024:.2f} KB)")
print(f"INT8 TFLite export:   {'✓' if INT8_OK else '❌'}")
if INT8_OK:
    print(f"INT8 size reduction:  {len(tflite_f32)/len(tflite_int8):.1f}x")
print(f"\nVerdict: {'✅ PROCEED with AgriYouthNet-Context+' if INT8_OK else '⚠️  NEED FALLBACK STRATEGY'}")