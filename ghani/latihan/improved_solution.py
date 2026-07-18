"""
Improved Solution - Neural Network Ensemble (Pure NumPy)
Pitri Workspace - Datathon 2026

Hasil validasi: ~62% accuracy (vs baseline ~54%)
Target 70-80%: gunakan catboost/lightgbm di Google Colab

Cara run:
  python3 improved_solution.py
  (Tidak perlu install apapun - hanya butuh numpy & pandas)
"""

import numpy as np
import pandas as pd
import time
import os

# ── Config ────────────────────────────────────────────────────────────────
np.random.seed(42)
t0 = time.time()

# Sesuaikan path jika beda environment
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
train = pd.read_csv(f'{BASE}/train.csv')
test  = pd.read_csv(f'{BASE}/test.csv')
print(f"Train: {train.shape}  Test: {test.shape}")
print("Target distribution:\n", train['target'].value_counts().sort_index())


# ── Feature Engineering ───────────────────────────────────────────────────
def create_features(df):
    """
    Advanced feature engineering.
    Temuan kunci: rasio (task completion) dan nsd (grade variability)
    adalah fitur paling prediktif (corr ~0.40 dengan target).
    """
    df = df.copy()
    nl = [c for c in df.columns if 'nilai_minggu' in c]   # 12 kolom nilai mingguan
    al = [c for c in df.columns if 'aktivitas_hari' in c] # 16 kolom aktivitas harian
    nv = df[nl].values.astype(float)
    av = df[al].values.astype(float)
    mn = len(nl) // 2
    ma = len(al) // 2

    # ── Statistik Nilai Mingguan ──────────────────────────────────────────
    df['nm']     = nv.mean(1)
    df['nsd']    = nv.std(1)               # ★ Fitur terpenting (corr=0.40)
    df['nrng']   = nv.max(1) - nv.min(1)  # ★ Fitur kuat (corr=0.37)
    df['nh1']    = nv[:, :mn].mean(1)     # rata-rata semester 1
    df['nh2']    = nv[:, mn:].mean(1)     # rata-rata semester 2
    df['nh2h1']  = df['nh2'] - df['nh1'] # tren semester
    df['nl3']    = nv[:, -3:].mean(1)     # rata-rata 3 minggu terakhir
    df['nf3']    = nv[:, :3].mean(1)      # rata-rata 3 minggu pertama
    df['npos']   = (nv > 0).sum(1)        # jumlah minggu nilai positif
    df['nneg']   = (nv < 0).sum(1)        # jumlah minggu nilai negatif
    df['niqr']   = np.percentile(nv, 75, 1) - np.percentile(nv, 25, 1)
    df['ncon']   = 1 - df['nsd'] / (df['nm'].abs() + 1e-6)  # konsistensi nilai
    df['nmom']   = nv[:, -1] - nv[:, -4]  # momentum (minggu terakhir vs -4)
    df['nsd2']   = df['nsd'] ** 2
    df['nrng2']  = df['nrng'] ** 2

    # ── Statistik Aktivitas Harian ────────────────────────────────────────
    df['am']     = av.mean(1)
    df['asd']    = av.std(1)
    df['arng']   = av.max(1) - av.min(1)
    df['ah1']    = av[:, :ma].mean(1)
    df['ah2']    = av[:, ma:].mean(1)
    df['al3']    = av[:, -3:].mean(1)
    df['atr']    = av[:, -1] - av[:, 0]   # tren aktivitas
    df['ahigh']  = (av > av.mean(1, keepdims=True)).sum(1)  # hari di atas rata-rata

    # ── Fitur Tugas ────────────────────────────────────────────────────────
    df['rasio']  = df['tugas_selesai'] / (df['tugas_diberikan'] + 1e-6)  # ★ corr=0.40
    df['rasio2'] = df['rasio'] ** 2
    df['surp']   = df['tugas_selesai'] - df['tugas_diberikan']

    # ── Interaksi Fitur Terpenting ────────────────────────────────────────
    df['rasio_nsd']   = df['rasio'] * df['nsd']   # interaksi 2 fitur terkuat
    df['rasio_nrng']  = df['rasio'] * df['nrng']
    df['nsd_am']      = df['nsd'] * df['am']
    df['rasio_am']    = df['rasio'] * df['am']
    df['rasio_ahigh'] = df['rasio'] * df['ahigh']
    df['nsd_ahigh']   = df['nsd'] * df['ahigh']

    # ── Skor Komposit ─────────────────────────────────────────────────────
    df['tot']  = (df['skor_motivasi'] + df['skor_kedisiplinan'] +
                  df['skor_tryout']   + df['skor_literasi'])
    df['sak']  = (df['skor_tryout'] * 0.4 + df['skor_literasi'] * 0.3 +
                  df['nm'] * 0.2 + df['rasio'] * 0.1)
    df['spe']  = (df['skor_motivasi'] * 0.35 + df['skor_kedisiplinan'] * 0.35 +
                  df['indeks_kehadiran'] * 0.3)
    df['stot'] = df['sak'] * 0.6 + df['spe'] * 0.4
    df['eng']  = df['indeks_kehadiran'] * df['skor_minat_belajar'] * df['am']
    df['lmom'] = df['nl3'] * df['al3']   # learning momentum
    df['teff'] = df['rasio'] * df['am']  # task efficiency

    # ── Fitur Interaksi Lain ──────────────────────────────────────────────
    df['md']    = df['skor_motivasi']     * df['skor_kedisiplinan']
    df['nr']    = df['nm']               * df['rasio']
    df['th']    = df['skor_tryout']      * df['indeks_kehadiran']
    df['an']    = df['am']               * df['nm']
    df['mn2']   = df['skor_motivasi']    * df['nm']
    df['tn']    = df['tot']              * df['nm']
    df['trn']   = df['skor_tryout']      * df['nm']
    df['lm']    = df['skor_literasi']    * df['skor_motivasi']
    df['sakam'] = df['sak']              * df['am']
    df['exm']   = df['skor_ekstrakurikuler'] * df['skor_motivasi']

    # ── Polynomial ────────────────────────────────────────────────────────
    df['tr2']   = df['skor_tryout'] ** 2
    df['ak2']   = df['sak'] ** 2
    df['a2']    = df['am'] ** 2

    # ── Binning ──────────────────────────────────────────────────────────
    df['kbin'] = pd.cut(df['kelas'], 10, labels=False).fillna(5).astype(int)
    df['ubin'] = pd.cut(df['urutan_ujian'], 10, labels=False).fillna(5).astype(int)

    return df.fillna(0).replace([np.inf, -np.inf], 0)


train_fe = create_features(train)
test_fe  = create_features(test)
drop_cols = {'id', 'target', 'kelas'}
feat_cols = [c for c in train_fe.columns if c not in drop_cols and c in test_fe.columns]

X  = train_fe[feat_cols].values.astype(np.float64)
y  = train_fe['target'].values.astype(int)
Xt = test_fe[feat_cols].values.astype(np.float64)

# Normalize (StandardScaler manual)
X_mu  = X.mean(0)
X_sig = X.std(0) + 1e-8
X  = (X - X_mu) / X_sig
Xt = (Xt - X_mu) / X_sig
X  = np.nan_to_num(X)
Xt = np.nan_to_num(Xt)
print(f"Features: {len(feat_cols)},  FE done in {time.time()-t0:.2f}s")

# ── Stratified Train/Val Split ────────────────────────────────────────────
val_idx = np.concatenate([
    np.random.choice(np.where(y == c)[0], int((y == c).sum() * 0.2), replace=False)
    for c in range(4)])
tr_idx = np.setdiff1d(np.arange(len(X)), val_idx)
Xtr, Xv = X[tr_idx], X[val_idx]
ytr, yv = y[tr_idx], y[val_idx]
print(f"Train: {len(Xtr)},  Val: {len(Xv)}")

yoh = np.eye(4)[ytr]   # one-hot labels
D   = Xtr.shape[1]


# ── Neural Network dengan Adam + Dropout ─────────────────────────────────
def train_model(seed, hidden_dims, lr, dropout, l2, epochs, batch_size=128):
    """
    Train MLP dengan:
    - He initialization
    - ReLU activations
    - Dropout regularization
    - L2 weight decay
    - Adam optimizer
    - Early stopping via checkpoint
    """
    np.random.seed(seed)
    dims  = [D] + hidden_dims + [4]
    n_lay = len(dims) - 1

    # Initialize weights (He)
    W = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2. / dims[i])
         for i in range(n_lay)]
    b = [np.zeros(dims[i+1]) for i in range(n_lay)]

    # Adam state
    mW = [np.zeros_like(w) for w in W]
    vW = [np.zeros_like(w) for w in W]
    mb = [np.zeros_like(x) for x in b]
    vb = [np.zeros_like(x) for x in b]
    step = 0

    best_acc = 0
    best_W   = None
    best_b   = None

    for ep in range(epochs):
        # Mini-batch SGD
        idx = np.random.permutation(len(Xtr))
        for s in range(0, len(Xtr), batch_size):
            bi  = idx[s:s + batch_size]
            xb  = Xtr[bi]
            yb  = yoh[bi]

            # Forward pass
            acts  = [xb]
            masks = []
            a = xb
            for i in range(n_lay - 1):
                a = np.maximum(0, a @ W[i] + b[i])
                if dropout > 0:
                    m = (np.random.rand(*a.shape) >= dropout) / (1 - dropout)
                    a = a * m
                    masks.append(m)
                else:
                    masks.append(None)
                acts.append(a)

            # Softmax output
            z  = a @ W[-1] + b[-1]
            z -= z.max(1, keepdims=True)
            ez = np.exp(z)
            p  = ez / ez.sum(1, keepdims=True)
            acts.append(p)

            # Backward pass
            dz   = (p - yb) / len(bi)
            step += 1
            c1   = 1 - 0.9  ** step
            c2   = 1 - 0.999 ** step

            for i in range(n_lay - 1, -1, -1):
                dW = acts[i].T @ dz + l2 * W[i]
                db2 = dz.sum(0)

                # Adam update
                mW[i] = 0.9 * mW[i] + 0.1 * dW
                vW[i] = 0.999 * vW[i] + 0.001 * dW ** 2
                mb[i] = 0.9 * mb[i] + 0.1 * db2
                vb[i] = 0.999 * vb[i] + 0.001 * db2 ** 2
                W[i] -= lr * (mW[i] / c1) / (np.sqrt(vW[i] / c2) + 1e-8)
                b[i] -= lr * (mb[i] / c1) / (np.sqrt(vb[i] / c2) + 1e-8)

                if i > 0:
                    da = dz @ W[i].T
                    if masks[i - 1] is not None:
                        da = da * masks[i - 1]
                    dz = da * (acts[i] > 0)

        # Checkpoint every 15 epochs
        if ep % 15 == 0 or ep == epochs - 1:
            a = Xv
            for i in range(n_lay - 1):
                a = np.maximum(0, a @ W[i] + b[i])
            z  = a @ W[-1] + b[-1]
            z -= z.max(1, keepdims=True)
            ez = np.exp(z)
            pv = ez / ez.sum(1, keepdims=True)
            acc = (pv.argmax(1) == yv).mean()
            if acc > best_acc:
                best_acc = acc
                best_W   = [w.copy() for w in W]
                best_b   = [bx.copy() for bx in b]

    # Restore best checkpoint
    if best_W:
        W, b = best_W, best_b

    def predict_proba(Xp):
        a = Xp
        for i in range(n_lay - 1):
            a = np.maximum(0, a @ W[i] + b[i])
        z  = a @ W[-1] + b[-1]
        z -= z.max(1, keepdims=True)
        ez = np.exp(z)
        return ez / ez.sum(1, keepdims=True)

    return predict_proba, best_acc


# ── Ensemble Training ─────────────────────────────────────────────────────
# 9 model dengan hyperparameter beragam
# Dropout tinggi (0.5-0.6) dan L2 kuat karena dataset kecil (3200 samples)
model_configs = [
    # (seed, hidden_dims,  lr,     dropout, l2,    epochs)
    (42,  [128, 64], 0.001,  0.5, 1e-3, 110),
    (99,  [128, 64], 0.001,  0.5, 1e-3, 110),
    (7,   [64,  32], 0.002,  0.5, 5e-4, 110),
    (13,  [64,  32], 0.002,  0.5, 5e-4, 110),
    (21,  [128, 64], 0.003,  0.6, 2e-3, 110),
    (77,  [256,128], 0.001,  0.5, 1e-3, 110),
    (11,  [128, 64], 0.001,  0.6, 5e-4, 110),
    (22,  [64,  32], 0.003,  0.5, 1e-3, 110),
    (33,  [128, 64], 0.002,  0.5, 2e-3, 110),
]

val_probs  = np.zeros((len(Xv),  4))
test_probs = np.zeros((len(Xt), 4))
best_single_acc = 0
best_test_probs = None

print(f"\n=== Training {len(model_configs)} Neural Networks ===")
for i, (seed, hidden, lr, dr, l2, ep) in enumerate(model_configs):
    pred_fn, acc = train_model(seed, hidden, lr, dr, l2, ep)
    vp  = pred_fn(Xv)
    tp  = pred_fn(Xt)
    val_probs  += vp
    test_probs += tp
    if acc > best_single_acc:
        best_single_acc = acc
        best_test_probs  = tp.copy()
    ens_acc = (val_probs.argmax(1) == yv).mean()
    print(f"  Model {i+1:2d} [{str(hidden):12s}] val={acc:.4f}  ens={ens_acc:.4f}  t={time.time()-t0:.0f}s")

ens_acc = (val_probs.argmax(1) == yv).mean()
print(f"\nBest single model : {best_single_acc:.4f}")
print(f"Ensemble accuracy : {ens_acc:.4f}")
print(f"Baseline (catboost): ~0.5397")
print(f"Improvement       : +{max(ens_acc, best_single_acc) - 0.5397:.4f}")
print(f"Total time        : {time.time()-t0:.1f}s")

# ── Generate Submission ───────────────────────────────────────────────────
# Gunakan best single model jika lebih baik dari ensemble
final_probs = test_probs if ens_acc >= best_single_acc else best_test_probs
final_label = "ensemble" if ens_acc >= best_single_acc else "best_single"
print(f"\nUsing: {final_label}")

out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'submission_nn_improved.csv')

submission = pd.DataFrame({'id': test['id'], 'target': final_probs.argmax(1)})
submission.to_csv(out_path, index=False)
print(f"Submission saved: {out_path}")
print("Prediction distribution:\n", submission['target'].value_counts().sort_index())
