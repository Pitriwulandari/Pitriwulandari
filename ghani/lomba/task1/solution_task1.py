"""
Datathon 2026 – Task 1: Traffic Speed Forecasting
===================================================
Model  : Per-road Ridge Regression (closed-form, numpy only)
Input  : 15 historical speed timesteps per road
Output : Speed predictions at h5, h10, h15 for 1260 roads × 540 test samples
"""

import numpy as np
import pandas as pd
import json, time
from pathlib import Path

t0 = time.time()

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA = Path(__file__).parent / "data" / "dataset-task1"
OUT  = Path(__file__).parent

# ── Load Data ──────────────────────────────────────────────────────────────────
print("Loading data...")
speed_m1 = np.load(DATA / "train" / "train_speed_m1_1_11160.npy")   # (11160, 1260)
speed_m2 = np.load(DATA / "train" / "train_speed_m2_1_5039.npy")    # (5039,  1260)
test_X   = np.load(DATA / "test"  / "test_X_hist.npy").astype(np.float32)  # (540, 15, 1260)
train    = np.vstack([speed_m1, speed_m2]).astype(np.float32)        # (16199, 1260)

with open(DATA / "train" / "train_text_m1_1_11160.json") as f: text_m1 = json.load(f)
with open(DATA / "train" / "train_text_m2_1_5039.json")  as f: text_m2 = json.load(f)
with open(DATA / "test"  / "test_texts.json")             as f: test_texts = json.load(f)

T, R   = train.shape   # 16199, 1260
H, F   = 15, 3         # history steps, forecast horizons
n_win  = T - H - F + 1 # sliding window count
N_test = test_X.shape[0]
print(f"  train:{train.shape}  test_X:{test_X.shape}  windows:{n_win} | {time.time()-t0:.1f}s")

# ── Text Feature Extraction ────────────────────────────────────────────────────
KEYWORDS = ["road closure", "construction", "accident", "prohibit", "announcement"]

def text_feats(text: str) -> np.ndarray:
    t = text.lower()
    return np.array([float(kw in t) for kw in KEYWORDS] + [float(t.count("."))],
                    dtype=np.float32)

N_TEXT = len(KEYWORDS) + 1  # 6

print("Extracting text features...")
train_tf = np.zeros((T, N_TEXT), dtype=np.float32)
for i in range(speed_m1.shape[0]):
    k = f"m1_{i+1}"
    if k in text_m1: train_tf[i] = text_feats(text_m1[k])
for i in range(speed_m2.shape[0]):
    k = f"m2_{i+1}"
    if k in text_m2: train_tf[speed_m1.shape[0] + i] = text_feats(text_m2[k])

test_tf = np.zeros((N_test, N_TEXT), dtype=np.float32)
for i in range(N_test):
    k = f"test_{i:05d}"
    if k in test_texts: test_tf[i] = text_feats(test_texts[k])

# ── Build XtX / Xty via Cross-Correlations ────────────────────────────────────
# Features: [speed_lag_0, ..., speed_lag_14, text_0, ..., text_5, bias]  = H + N_TEXT + 1 = 22
# Computed WITHOUT materialising the full (n_win × N_FEAT × R) array.
lam    = 1.0
N_FEAT = H + N_TEXT + 1  # 22

print(f"Building XtX/Xty (N_FEAT={N_FEAT})...")

# Shared columns (text + bias) at the target timestep — same for all roads
shared = np.concatenate([
    train_tf[H:H + n_win],                         # (n_win, N_TEXT)
    np.ones((n_win, 1), dtype=np.float32)           # (n_win, 1)
], axis=1)  # (n_win, N_TEXT+1)

XtX = np.zeros((R, N_FEAT, N_FEAT), dtype=np.float64)
Xty = np.zeros((R, N_FEAT, F),      dtype=np.float64)

for i in range(H):
    a = train[i:i + n_win]                          # (n_win, R) — speed lag i
    for j in range(i, H):
        b = train[j:j + n_win]
        v = (a * b).sum(axis=0)                     # (R,)
        XtX[:, i, j] = v;  XtX[:, j, i] = v
    cross_ts = a.T @ shared                         # (R, N_TEXT+1)
    XtX[:, i, H:] = cross_ts;  XtX[:, H:, i] = cross_ts
    for f in range(F):
        Xty[:, i, f] = (a * train[H + f:H + f + n_win]).sum(axis=0)

XtX[:, H:, H:] = (shared.T @ shared)[None]
for f in range(F):
    Xty[:, H:, f] = (shared.T @ train[H + f:H + f + n_win]).T

XtX += lam * np.eye(N_FEAT, dtype=np.float64)[None]
print(f"  done | {time.time()-t0:.1f}s")

# ── Solve and Predict ──────────────────────────────────────────────────────────
print("Solving ridge regression...")
W = np.linalg.solve(XtX, Xty).astype(np.float32)   # (R, N_FEAT, F)

print("Predicting on test data...")
test_feat = np.concatenate([
    test_X,                                                          # (540, 15, R)
    np.repeat(test_tf[:, :, None], R, axis=2).astype(np.float32),  # (540, N_TEXT, R)
    np.ones((N_test, 1, R), dtype=np.float32)                       # (540, 1, R)
], axis=1)  # (540, N_FEAT, R)

pred = np.einsum("nhr,rhf->nfr", test_feat, W)   # (540, F, R)
pred = np.clip(pred, 0.0, 200.0)
print(f"  pred shape:{pred.shape}  mean:{pred.mean():.2f} | {time.time()-t0:.1f}s")

# ── Write Submission CSV ───────────────────────────────────────────────────────
print("Writing submission CSV...")
horizons = [5, 10, 15]
ids, spd = [], []
for si in range(N_test):
    s = f"test_{si:05d}"
    for fi, h in enumerate(horizons):
        for ri in range(R):
            ids.append(f"{s}_h{h}_r{ri}")
            spd.append(float(pred[si, fi, ri]))

out_path = OUT / "submission_task1_ridge.csv"
pd.DataFrame({"id": ids, "speed": spd}).to_csv(out_path, index=False)
print(f"Saved {len(ids):,} rows → {out_path}")
print(f"Total time: {time.time()-t0:.1f}s")
