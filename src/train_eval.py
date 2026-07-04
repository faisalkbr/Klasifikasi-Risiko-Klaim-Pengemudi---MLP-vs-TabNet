"""
Pipeline lengkap: preprocessing -> training (MLP & TabNet) -> evaluasi -> ekspor.
Studi kasus: Klasifikasi Risiko Klaim Pengemudi (telematics asuransi).
Target: ClaimYN = 1 jika NB_Claim >= 1 (pengemudi pernah mengajukan klaim).

Jalankan:  python3 src/train_eval.py            (full)
           SMOKE=1 python3 src/train_eval.py    (cepat, untuk uji pipeline)
"""
import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
import sys
sys.path.append(os.path.dirname(__file__))
from models import build_mlp, build_tabnet

SMOKE = os.environ.get("SMOKE") == "1"
SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "telematics.csv")
FIG  = os.path.join(ROOT, "outputs", "figures")
MOD  = os.path.join(ROOT, "outputs", "models")
os.makedirs(FIG, exist_ok=True); os.makedirs(MOD, exist_ok=True)

sns.set_style("whitegrid")
PALETTE = {"MLP": "#2563eb", "TabNet": "#16a34a"}


# ---------------------------------------------------------------- 1. LOAD
df = pd.read_csv(DATA)
if SMOKE:
    df = df.sample(6000, random_state=SEED).reset_index(drop=True)
df["ClaimYN"] = (df["NB_Claim"] >= 1).astype(int)

TARGET = "ClaimYN"
LEAK   = ["NB_Claim", "AMT_Claim"]          # WAJIB dibuang: mendefinisikan target
cat_cols = ["Insured.sex", "Marital", "Car.use", "Region"]
num_cols = [c for c in df.columns if c not in cat_cols + LEAK + [TARGET]]

print(f"[DATA] rows={len(df)}  positives={df[TARGET].sum()} "
      f"({100*df[TARGET].mean():.2f}%)  num={len(num_cols)} cat={len(cat_cols)}")

# ---------------------------------------------------------------- 2. ENCODE
df_enc = pd.get_dummies(df[cat_cols], prefix=cat_cols)
cat_feat = df_enc.columns.tolist()
X_all = pd.concat([df[num_cols], df_enc], axis=1).astype("float32")
y_all = df[TARGET].values
feature_names = X_all.columns.tolist()

# ---------------------------------------------------------------- 3. SPLIT 70/15/15
X_tr, X_tmp, y_tr, y_tmp = train_test_split(
    X_all, y_all, test_size=0.30, stratify=y_all, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)

# ---------------------------------------------------------------- 4. SCALE (fit di train saja)
scaler = StandardScaler().fit(X_tr[num_cols])
def scale(Xdf):
    X = Xdf.copy()
    X[num_cols] = scaler.transform(X[num_cols])
    return X.values.astype("float32")
Xtr, Xva, Xte = scale(X_tr), scale(X_va), scale(X_te)
print(f"[SPLIT] train={len(Xtr)} val={len(Xva)} test={len(Xte)}")

# ---------------------------------------------------------------- 5. CLASS WEIGHT (imbalance)
cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
class_weight = {0: float(cw[0]), 1: float(cw[1])}
print(f"[IMBALANCE] class_weight={class_weight}")

EPOCHS = 3 if SMOKE else 80
BATCH  = 512
INPUT_DIM = Xtr.shape[1]

def train_model(builder, name):
    tf.keras.backend.clear_session()
    tf.random.set_seed(SEED)
    model = builder(INPUT_DIM)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    es = EarlyStopping(monitor="val_auc", mode="max", patience=10,
                       restore_best_weights=True, verbose=0)
    hist = model.fit(Xtr, y_tr, validation_data=(Xva, y_va),
                     epochs=EPOCHS, batch_size=BATCH, class_weight=class_weight,
                     callbacks=[es], verbose=0)
    print(f"[TRAIN] {name}: {len(hist.history['loss'])} epochs, "
          f"best val_auc={max(hist.history['val_auc']):.4f}")
    return model, hist.history

def best_threshold(model):
    p = model.predict(Xva, verbose=0).ravel()
    prec, rec, thr = precision_recall_curve(y_va, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(thr[np.argmax(f1[:-1])])

def evaluate(model, thr, name):
    p = model.predict(Xte, verbose=0).ravel()
    yp = (p >= thr).astype(int)
    m = dict(
        accuracy=accuracy_score(y_te, yp),
        precision=precision_score(y_te, yp, zero_division=0),
        recall=recall_score(y_te, yp, zero_division=0),
        f1=f1_score(y_te, yp, zero_division=0),
        auc=roc_auc_score(y_te, p),
        mcc=matthews_corrcoef(y_te, yp),
        threshold=thr)
    return m, p, yp


# =============================================================== RUN
results, histories, probs, preds = {}, {}, {}, {}
models = {}
for builder, name in [(build_mlp, "MLP"), (build_tabnet, "TabNet")]:
    model, hist = train_model(builder, name)
    thr = best_threshold(model)
    met, p, yp = evaluate(model, thr, name)
    results[name] = met; histories[name] = hist; probs[name] = p; preds[name] = yp
    models[name] = model
    model.save(os.path.join(MOD, f"{name}.keras"))
    print(f"[EVAL ] {name}: F1={met['f1']:.3f} Recall={met['recall']:.3f} "
          f"AUC={met['auc']:.3f} MCC={met['mcc']:.3f} thr={thr:.3f}")

# ---------------------------------------------------------------- FIGURES
# (a) training curves per model: accuracy + loss
for name, h in histories.items():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(h["accuracy"], label="Train", color=PALETTE[name])
    ax[0].plot(h["val_accuracy"], label="Validation", color="#f59e0b")
    ax[0].set_title(f"Model Accuracy - {name}"); ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy"); ax[0].legend()
    ax[1].plot(h["loss"], label="Train Loss", color=PALETTE[name])
    ax[1].plot(h["val_loss"], label="Validation Loss", color="#f59e0b")
    ax[1].set_title(f"Model Loss - {name}"); ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss"); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"training_{name}.png"), dpi=140)
    plt.close()

# (b) confusion matrices
for name in results:
    cm = confusion_matrix(y_te, preds[name])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Tdk Berisiko", "Berisiko"],
                yticklabels=["Tdk Berisiko", "Berisiko"], ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, f"confusion_{name}.png"), dpi=140)
    plt.close()

# (c) ROC comparison
fig, ax = plt.subplots(figsize=(5.5, 5))
for name in results:
    fpr, tpr, _ = roc_curve(y_te, probs[name])
    ax.plot(fpr, tpr, color=PALETTE[name],
            label=f"{name} (AUC={results[name]['auc']:.3f})")
ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - MLP vs TabNet"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "roc_comparison.png"), dpi=140); plt.close()

# (d) PR comparison
fig, ax = plt.subplots(figsize=(5.5, 5))
for name in results:
    prec, rec, _ = precision_recall_curve(y_te, probs[name])
    ax.plot(rec, prec, color=PALETTE[name], label=name)
ax.axhline(y_te.mean(), ls="--", color="grey", label=f"Baseline ({y_te.mean():.3f})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "pr_comparison.png"), dpi=140); plt.close()

# (e) metrics comparison bar
metrics_order = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]
fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(metrics_order)); w = 0.38
for i, name in enumerate(results):
    vals = [results[name][k] for k in metrics_order]
    b = ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=PALETTE[name])
    ax.bar_label(b, fmt="%.2f", fontsize=8, padding=2)
ax.set_xticks(x); ax.set_xticklabels([m.upper() for m in metrics_order])
ax.set_ylim(0, 1.05); ax.set_title("Perbandingan Metrik - MLP vs TabNet"); ax.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG, "metrics_comparison.png"), dpi=140); plt.close()

# (f) target distribution (EDA)
fig, ax = plt.subplots(figsize=(5, 4))
vc = pd.Series(y_all).value_counts().sort_index()
b = ax.bar(["Tidak Berisiko (0)", "Berisiko (1)"], vc.values,
           color=["#94a3b8", "#dc2626"])
ax.bar_label(b, fmt="%d"); ax.set_title("Distribusi Kelas Target (Imbalanced)")
ax.set_ylabel("Jumlah Sampel")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "target_distribution.png"), dpi=140); plt.close()

# ---------------------------------------------------------------- SAVE ARTIFACTS
with open(os.path.join(MOD, "metrics.json"), "w") as f:
    json.dump({"results": results,
               "class_balance": {"positive": int(y_all.sum()),
                                 "total": int(len(y_all)),
                                 "pct": float(y_all.mean())},
               "n_features": INPUT_DIM,
               "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)}},
              f, indent=2)

# preprocessing spec untuk Flutter (mean/scale + urutan fitur + mapping kategori)
prep = {
    "feature_order": feature_names,
    "numeric_features": num_cols,
    "categorical_onehot": cat_feat,
    "scaler_mean": dict(zip(num_cols, scaler.mean_.tolist())),
    "scaler_scale": dict(zip(num_cols, scaler.scale_.tolist())),
    "categorical_values": {c: sorted(df[c].unique().tolist()) for c in cat_cols},
    "best_model": max(results, key=lambda k: results[k]["f1"]),
    "thresholds": {k: results[k]["threshold"] for k in results},
}
with open(os.path.join(MOD, "preprocessing.json"), "w") as f:
    json.dump(prep, f, indent=2)

# ---------------------------------------------------------------- TFLITE EXPORT
for name, model in models.items():
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl = conv.convert()
    with open(os.path.join(MOD, f"{name}.tflite"), "wb") as f:
        f.write(tfl)
    print(f"[TFLITE] {name}.tflite = {len(tfl)/1024:.1f} KB")

# sample prediksi untuk laporan
sample = pd.DataFrame({
    "aktual": y_te[:15],
    "prob_MLP": probs["MLP"][:15].round(3),
    "pred_MLP": preds["MLP"][:15],
    "prob_TabNet": probs["TabNet"][:15].round(3),
    "pred_TabNet": preds["TabNet"][:15],
})
sample.to_csv(os.path.join(MOD, "sample_predictions.csv"), index=False)

print("\n=== COMPARISON TABLE ===")
print(f"{'Metric':<12}{'MLP':>10}{'TabNet':>10}")
for k in metrics_order:
    print(f"{k:<12}{results['MLP'][k]:>10.4f}{results['TabNet'][k]:>10.4f}")
print(f"\nBest model (by F1): {prep['best_model']}")
print("DONE")
