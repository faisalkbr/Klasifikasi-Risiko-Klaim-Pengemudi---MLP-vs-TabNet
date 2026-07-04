"""
Definisi arsitektur dua metode Deep Learning untuk klasifikasi risiko klaim.
Method 1: Deep MLP (Feedforward Neural Network)
Method 2: TabNet-style network (GLU feature transformer + sequential attention)

Keduanya dibangun murni di tf.keras => bisa diekspor ke TensorFlow Lite (.tflite)
untuk inferensi on-device di aplikasi Flutter.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Input, regularizers


# ---------------------------------------------------------------------------
# METHOD 1 : Deep MLP
# ---------------------------------------------------------------------------
def build_mlp(input_dim, l2=1e-4):
    inp = Input(shape=(input_dim,), name="features")
    x = layers.BatchNormalization()(inp)
    for units in (128, 64, 32):
        x = layers.Dense(units, activation="relu",
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="risk")(x)
    return Model(inp, out, name="MLP")


# ---------------------------------------------------------------------------
# METHOD 2 : TabNet-style network (TFLite-friendly, softmax attention)
# ---------------------------------------------------------------------------
def _glu_block(x, units):
    """Gated Linear Unit block: inti dari Feature Transformer TabNet."""
    linear = layers.Dense(units)(x)
    gate = layers.Dense(units, activation="sigmoid")(x)
    return layers.Multiply()([linear, gate])


def build_tabnet(input_dim, feature_dim=32, n_steps=3, l2=1e-4):
    """
    TabNet sederhana yang mempertahankan ide utama paper:
      - pemrosesan multi-step berurutan (sequential decision steps)
      - attentive transformer: memilih fitur paling relevan tiap step (mask)
      - feature transformer berbasis GLU
    Mask memakai softmax (bukan sparsemax) agar kompatibel penuh dengan TFLite.
    """
    inp = Input(shape=(input_dim,), name="features")
    x = layers.BatchNormalization()(inp)

    prior = layers.Lambda(lambda t: tf.ones_like(t),
                          name="prior_init")(x)   # prior scale awal = 1 tiap fitur
    decision_agg = None

    for step in range(n_steps):
        # --- Attentive Transformer: hasilkan mask seleksi fitur ---
        att = layers.Dense(input_dim)(x)
        att = layers.BatchNormalization()(att)
        att = layers.Multiply()([att, prior])          # tekankan prior
        mask = layers.Softmax(name=f"mask_{step}")(att)  # bobot fitur (jumlah=1)

        # perbarui prior: fitur yang sudah dipakai diberi bobot lebih kecil
        prior = layers.Multiply()([prior, layers.Lambda(lambda m: 1.5 - m)(mask)])

        # --- terapkan mask ke input, lalu Feature Transformer (GLU) ---
        masked = layers.Multiply()([inp, mask])
        ft = _glu_block(masked, feature_dim)
        ft = layers.BatchNormalization()(ft)
        ft = _glu_block(ft, feature_dim)

        # output keputusan step ini
        decision = layers.Activation("relu")(ft)
        decision_agg = decision if decision_agg is None else layers.Add()([decision_agg, decision])

        # info yang diteruskan ke step berikutnya
        x = layers.Dense(feature_dim, activation="relu")(ft)

    out = layers.Dense(1, activation="sigmoid", name="risk")(decision_agg)
    return Model(inp, out, name="TabNet")


if __name__ == "__main__":
    import numpy as np, tempfile, os
    D = 20
    for builder in (build_mlp, build_tabnet):
        m = builder(D)
        m.compile(optimizer="adam", loss="binary_crossentropy")
        xb = np.random.rand(8, D).astype("float32")
        _ = m.predict(xb, verbose=0)
        # cek konversi TFLite
        conv = tf.lite.TFLiteConverter.from_keras_model(m)
        tfl = conv.convert()
        print(f"{m.name:8s} | params={m.count_params():>7,} | forward OK | TFLite {len(tfl)/1024:.1f} KB")
