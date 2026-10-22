"""
AgriYouthNet-Context+: Ultra-Lightweight Cross-Attention Fusion
for Mobile Plant Disease Diagnosis.

Wide configuration (~140K params) matches the proposal's target.
Supports:
  - fusion="attention": MHCA with num_heads heads
  - fusion="concat": fixed-weight concatenation (baseline)
  - fusion="gating": sigmoid-gated fusion (baseline)
  - width="wide" | "narrow"
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Custom layers (serializable — no Lambdas)
# ─────────────────────────────────────────────────────────────────
class MultiHeadCrossAttention(layers.Layer):
    """MHCA: queries attend to keys/values from another modality."""

    def __init__(self, num_heads: int = 4, key_dim: int = 32,
                 out_dim: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.out_dim = out_dim
        self.scale = float(key_dim) ** -0.5

    def build(self, input_shape):
        q_shape, kv_shape = input_shape
        if self.out_dim is None:
            self.out_dim = q_shape[-1]
        self.q_proj = layers.Dense(self.num_heads * self.key_dim,
                                   use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(self.num_heads * self.key_dim,
                                   use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(self.num_heads * self.key_dim,
                                   use_bias=False, name="v_proj")
        self.out_proj = layers.Dense(self.out_dim,
                                     use_bias=False, name="out_proj")
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, kv = inputs
        q_len = query.shape[1]
        kv_len = kv.shape[1]
        H, D = self.num_heads, self.key_dim

        q = self.q_proj(query)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        q = tf.transpose(tf.reshape(q, (-1, q_len, H, D)), (0, 2, 1, 3))
        k = tf.transpose(tf.reshape(k, (-1, kv_len, H, D)), (0, 2, 1, 3))
        v = tf.transpose(tf.reshape(v, (-1, kv_len, H, D)), (0, 2, 1, 3))

        scores = tf.matmul(q, k, transpose_b=True) * self.scale
        weights = tf.nn.softmax(scores, axis=-1)

        out = tf.matmul(weights, v)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (-1, q_len, H * D))
        out = self.out_proj(out)
        return out, weights

    def compute_output_shape(self, input_shape):
        q_shape, kv_shape = input_shape
        return [
            (q_shape[0], q_shape[1], self.out_dim),
            (q_shape[0], self.num_heads, q_shape[1], kv_shape[1]),
        ]

    def get_config(self):
        c = super().get_config()
        c.update({"num_heads": self.num_heads,
                  "key_dim": self.key_dim,
                  "out_dim": self.out_dim})
        return c


class MeanOverHeads(layers.Layer):
    """Reduce (B, H, 1, context_dim) → (B, 1, context_dim).
    Serializable replacement for a Python lambda."""

    def call(self, x):
        return tf.reduce_mean(x, axis=1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1, input_shape[3])


# ─────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────
def _depthwise_separable(x, filters: int, stride: int, name: str):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                               use_bias=False, name=f"{name}_dw")(x)
    x = layers.BatchNormalization(name=f"{name}_dw_bn")(x)
    x = layers.ReLU(name=f"{name}_dw_relu")(x)
    x = layers.Conv2D(filters, 1, padding="same",
                      use_bias=False, name=f"{name}_pw")(x)
    x = layers.BatchNormalization(name=f"{name}_pw_bn")(x)
    x = layers.ReLU(name=f"{name}_pw_relu")(x)
    return x


def _visual_backbone(image_input, width: str = "wide",
                     name_prefix: str = "backbone"):
    """
    AgriYouthNet visual backbone.
      width="narrow" → c1=32, c2=64, c3=128    (output dim 128)
      width="wide"   → c1=64, c2=128, c3=256   (output dim 256)
    """
    if width == "narrow":
        c1, c2, c3 = 32, 64, 128
    elif width == "wide":
        c1, c2, c3 = 64, 128, 256
    else:
        raise ValueError(f"Unknown width: {width} (use 'narrow' or 'wide')")

    x = layers.Conv2D(c1, 3, strides=2, padding="same",
                      use_bias=False, name=f"{name_prefix}_conv1")(image_input)
    x = layers.BatchNormalization(name=f"{name_prefix}_conv1_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_conv1_relu")(x)

    x = _depthwise_separable(x, c2, stride=2, name=f"{name_prefix}_conv2")
    x = _depthwise_separable(x, c3, stride=2, name=f"{name_prefix}_conv3")

    x = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(x)
    return x, c3


def _context_encoder(context_input, context_dim: int, token_dim: int,
                     name_prefix: str = "context"):
    x = layers.Dense(context_dim * token_dim, activation="relu",
                     name=f"{name_prefix}_fc1")(context_input)
    x = layers.Dense(context_dim * token_dim, activation="relu",
                     name=f"{name_prefix}_fc2")(x)
    x = layers.Reshape((context_dim, token_dim),
                       name=f"{name_prefix}_tokens")(x)
    return x


# ─────────────────────────────────────────────────────────────────
# Main factory
# ─────────────────────────────────────────────────────────────────
def build_agriyouthnet_context_plus(
    num_heads: int = 4,
    key_dim: int = 32,
    context_dim: int = 4,
    num_classes: int = 10,
    input_size: int = 224,
    fusion: str = "attention",
    token_dim: int = 32,
    width: str = "wide",
    dropout: float = 0.1,
    name: str = "agriyouthnet_context_plus",
) -> Model:
    """
    Build AgriYouthNet-Context+ or an ablation variant.

    Args:
        num_heads:  attention heads (1, 2, 4, 8)
        key_dim:    per-head key dimension
        context_dim: length of context vector (4 for crop one-hot)
        num_classes: 10
        input_size:  224
        fusion:     "attention" | "concat" | "gating"
        token_dim:  context token dimension (default 32)
        width:      "wide" (140K params) | "narrow" (52K params)
        dropout:    classifier dropout rate
        name:       model name

    Returns:
        Keras Model with two outputs:
          - class_probs:       (B, num_classes)
          - attention_weights: (B, context_dim)
    """
    image_input = layers.Input(shape=(input_size, input_size, 3),
                               name="leaf_image")
    context_input = layers.Input(shape=(context_dim,),
                                 name="context_vector")

    visual_feat, visual_dim = _visual_backbone(image_input, width=width)
    context_tokens = _context_encoder(context_input, context_dim, token_dim)

    if fusion == "attention":
        vis_query = layers.Reshape((1, visual_dim),
                                   name="visual_query")(visual_feat)
        mhca = MultiHeadCrossAttention(num_heads=num_heads, key_dim=key_dim,
                                       out_dim=visual_dim, name="mhca")
        attended, attn_weights = mhca([vis_query, context_tokens])
        attended = layers.Reshape((visual_dim,),
                                  name="attended_flat")(attended)

        fused = layers.Add(name="fusion_add")([visual_feat, attended])
        fused = layers.LayerNormalization(name="fusion_ln")(fused)

        attn_mean = MeanOverHeads(name="attn_mean_over_heads")(attn_weights)
        attn_out = layers.Reshape((context_dim,),
                                  name="attention_weights")(attn_mean)

    elif fusion == "concat":
        ctx_flat = layers.GlobalAveragePooling1D(
            name="context_gap")(context_tokens)
        ctx_proj = layers.Dense(visual_dim, use_bias=False,
                                name="context_proj")(ctx_flat)
        fused = layers.Concatenate(name="concat_fusion")(
            [visual_feat, ctx_proj])
        fused = layers.Dense(visual_dim, use_bias=False,
                             name="concat_proj")(fused)
        fused = layers.LayerNormalization(name="fusion_ln")(fused)
        attn_out = layers.Dense(context_dim, activation="softmax",
                                name="attention_weights")(fused)

    elif fusion == "gating":
        ctx_flat = layers.GlobalAveragePooling1D(
            name="context_gap")(context_tokens)
        ctx_proj = layers.Dense(visual_dim, use_bias=False,
                                name="context_proj")(ctx_flat)
        gate = layers.Dense(visual_dim, activation="sigmoid",
                            use_bias=False, name="gate")(visual_feat)
        fused = layers.Add(name="gating_fusion")([
            gate * visual_feat,
            (1.0 - gate) * ctx_proj,
        ])
        fused = layers.LayerNormalization(name="fusion_ln")(fused)
        attn_out = layers.Dense(context_dim, activation="softmax",
                                name="attention_weights")(fused)

    else:
        raise ValueError(f"Unknown fusion: {fusion}")

    fused = layers.Dropout(dropout, name="classifier_dropout")(fused)
    probs = layers.Dense(num_classes, activation="softmax",
                         name="class_probs")(fused)

    return Model(inputs=[image_input, context_input],
                 outputs=[probs, attn_out], name=name)


# ─────────────────────────────────────────────────────────────────
# Convenience wrappers
# ─────────────────────────────────────────────────────────────────
def build_agriyouthnet_context(num_heads: int = 1,
                               context_dim: int = 4,
                               num_classes: int = 10,
                               input_size: int = 224,
                               width: str = "wide",
                               name: str = "agriyouthnet_context") -> Model:
    """Concatenation-fusion baseline (Kakooza 2026 style)."""
    return build_agriyouthnet_context_plus(
        num_heads=num_heads, context_dim=context_dim,
        num_classes=num_classes, input_size=input_size,
        fusion="concat", width=width, name=name,
    )


def build_agriyouthnet_no_context(num_classes: int = 10,
                                  input_size: int = 224,
                                  width: str = "wide",
                                  name: str = "agriyouthnet_no_context") -> Model:
    """Visual-only baseline."""
    image_input = layers.Input(shape=(input_size, input_size, 3),
                               name="leaf_image")
    visual_feat, _ = _visual_backbone(image_input, width=width)
    visual_feat = layers.Dropout(0.1)(visual_feat)
    probs = layers.Dense(num_classes, activation="softmax",
                         name="class_probs")(visual_feat)
    return Model(inputs=image_input, outputs=probs, name=name)


# ─────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("AgriYouthNet-Context+ — Architecture Verification")
    print("=" * 60)

    configs = [
        ("Primary (4h, wide)",   dict(num_heads=4, fusion="attention", width="wide")),
        ("Primary (4h, narrow)", dict(num_heads=4, fusion="attention", width="narrow")),
        ("Ablation: 1h wide",    dict(num_heads=1, fusion="attention", width="wide")),
        ("Ablation: 8h wide",    dict(num_heads=8, fusion="attention", width="wide")),
        ("Baseline: concat",     dict(num_heads=1, fusion="concat",    width="wide")),
        ("Baseline: gating",     dict(num_heads=1, fusion="gating",    width="wide")),
    ]

    print("\nConfig                     Params      F32 (KB)  INT8 est (KB)")
    print("-" * 65)
    primary = None
    for label, kwargs in configs:
        m = build_agriyouthnet_context_plus(**kwargs)
        n = m.count_params()
        if "Primary (4h, wide)" in label:
            primary = n
        print(f"{label:<26} {n:>8,}   {n*4/1024:>8.2f}   {n/1024:>10.2f}")
        del m

    print(f"\nPrimary params: {primary:,}  (proposal target ~150K)")
    if 100_000 <= primary <= 250_000:
        print("✅ Within target range")
    else:
        print("⚠️  Outside target range")

    # Functional test
    import numpy as np
    print("\nFunctional test...")
    m = build_agriyouthnet_context_plus(num_heads=4, fusion="attention", width="wide")
    img = np.random.rand(2, 224, 224, 3).astype(np.float32)
    ctx = np.eye(4, dtype=np.float32)[[0, 2]]
    probs, attn = m.predict([img, ctx], verbose=0)
    assert probs.shape == (2, 10), f"probs shape {probs.shape}"
    assert attn.shape == (2, 4), f"attn shape {attn.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(attn.sum(axis=1), 1.0, atol=1e-5)
    print("✅ Functional test passed.")