# !pip install -q shap

import copy, json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# from google.colab import drive
# drive.mount("/content/drive", force_remount=False)

# ROOT_DIR      = "/content/drive/MyDrive/CompNeuroscience-P1"
# FEATURES_DIR  = f"{ROOT_DIR}/lamem_features"
# FEATURES_DIR2 = f"{ROOT_DIR}/lamem_features2"
# SAVE_DIR      = f"{ROOT_DIR}/va_regression"


FEATURES_DIR  = f"/Users/celsocukier/Documents/CompNeuroscience/MemorabilityEmotions/lamem_features"
FEATURES_DIR2 = f"/Users/celsocukier/Documents/CompNeuroscience/MemorabilityEmotions/lamem_features2"
SAVE_DIR      = f"/Users/celsocukier/Documents/CompNeuroscience/MemorabilityEmotions/va_regression"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── 1 · Load VAMOS data ───────────────────────────────────────────────────────

VAMOS_CSV   = f"/Users/celsocukier/Documents/CompNeuroscience/MemorabilityEmotions/VAMOS_Set_1_img_info.csv"
VAMOS_EMB   = f"/Users/celsocukier/Documents/CompNeuroscience/MemorabilityEmotions/memcat_vit_results/clip_embeddings.npy"   # ← your 900-image .npy

vamos_df  = pd.read_csv(VAMOS_CSV)
vamos_emb = np.load(VAMOS_EMB).astype(np.float32)       # (900, 768)

assert len(vamos_df) == len(vamos_emb), \
    f"CSV rows ({len(vamos_df)}) ≠ embedding rows ({len(vamos_emb)})"

# Normalise VA from [1, 9] → [-1, 1]
va_raw = vamos_df[["Mean_Valence", "Mean_Arousal"]].values.astype(np.float32)
va     = ((va_raw - 5.0) / 4.0).astype(np.float32)      # (900, 2)
mem    = vamos_df["Memorability_Score"].values.astype(np.float32)

print(f"VAMOS  : {len(vamos_df)} images | sets: {vamos_df['Image_Set'].value_counts().to_dict()}")
print(f"Valence: [{va[:,0].min():.2f}, {va[:,0].max():.2f}]  "
      f"Arousal: [{va[:,1].min():.2f}, {va[:,1].max():.2f}]")

# ── 2 · VA distribution + VA-memorability scatter ────────────────────────────

fig = plt.figure(figsize=(15, 4))
gs  = gridspec.GridSpec(1, 3, figure=fig)

ax0 = fig.add_subplot(gs[0])
ax0.scatter(va_raw[:,0], va_raw[:,1], alpha=0.5, s=20,
            c=mem, cmap="RdYlGn", vmin=0.4, vmax=0.9)
ax0.set_xlabel("Valence (1–9)"); ax0.set_ylabel("Arousal (1–9)")
ax0.set_title("VA space (colour = memorability)")
ax0.axvline(5, color="gray", linestyle="--", linewidth=0.8)
ax0.axhline(5, color="gray", linestyle="--", linewidth=0.8)

for ax, dim, label in [(fig.add_subplot(gs[1]), 0, "Valence"),
                        (fig.add_subplot(gs[2]), 1, "Arousal")]:
    r, _ = pearsonr(va_raw[:,dim], mem)
    ax.scatter(va_raw[:,dim], mem, alpha=0.4, s=15, color="steelblue")
    m, b  = np.polyfit(va_raw[:,dim], mem, 1)
    xs    = np.linspace(va_raw[:,dim].min(), va_raw[:,dim].max(), 100)
    ax.plot(xs, m*xs+b, color="red", linewidth=1.5)
    ax.set_xlabel(f"{label} (1–9)"); ax.set_ylabel("Memorability")
    ax.set_title(f"{label} vs Memorability  r={r:.3f}")
    ax.grid(alpha=0.3)

plt.suptitle("VAMOS dataset — VA space and memorability correlation", fontsize=12)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/vamos_va_overview.png", dpi=150)
plt.show()

# ── 3 · CCC loss ─────────────────────────────────────────────────────────────

def ccc_loss(pred, target):
    pred_m   = pred.mean();   target_m   = target.mean()
    pred_v   = pred.var();    target_v   = target.var()
    cov      = ((pred - pred_m) * (target - target_m)).mean()
    ccc      = 2 * cov / (pred_v + target_v + (pred_m - target_m)**2 + 1e-8)
    return 1.0 - ccc

def combined_loss(pred, target, alpha=0.5):
    return alpha * ccc_loss(pred, target) + (1 - alpha) * F.mse_loss(pred, target)

def ccc_score(pred, target):
    pred_m   = pred.mean();   target_m   = target.mean()
    pred_v   = pred.var();    target_v   = target.var()
    cov      = ((pred - pred_m) * (target - target_m)).mean()
    return float(2 * cov / (pred_v + target_v + (pred_m - target_m)**2 + 1e-8))

# ── 4 · Model ─────────────────────────────────────────────────────────────────

class VAHead(nn.Module):
    def __init__(self, input_dim=768, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
            nn.Tanh(),              # output in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

# ── 5 · Dataset ───────────────────────────────────────────────────────────────

class VADataset(Dataset):
    def __init__(self, emb, va):
        self.x = torch.tensor(emb, dtype=torch.float32)
        self.y = torch.tensor(va,  dtype=torch.float32)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ── 6 · Train / eval helpers ─────────────────────────────────────────────────

def evaluate_va(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(DEVICE)).cpu())
            targets.append(y)
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()

    metrics = {}
    for i, dim in enumerate(["valence", "arousal"]):
        p, t       = preds[:,i], targets[:,i]
        r, _       = pearsonr(p, t)
        mse        = float(np.mean((p - t)**2))
        ccc        = ccc_score(torch.tensor(p), torch.tensor(t))
        metrics[dim] = {"r": round(r,4), "mse": round(mse,5), "ccc": round(ccc,4)}
    return preds, targets, metrics


def train_va(emb_train, va_train, emb_val, va_val,
             lr=5e-4, weight_decay=1e-4, max_epochs=300,
             patience=30, batch_size=64, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)

    tr_loader  = DataLoader(VADataset(emb_train, va_train),
                            batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VADataset(emb_val, va_val),
                            batch_size=batch_size, shuffle=False)
    model = VAHead().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    best_val_loss = float("inf")
    best_state    = None
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in tr_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred = model(x)
            loss = combined_loss(pred[:,0], y[:,0]) + combined_loss(pred[:,1], y[:,1])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        _, _, val_m = evaluate_va(model, val_loader)
        val_loss = -(val_m["valence"]["ccc"] + val_m["arousal"]["ccc"]) / 2

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    return model

# ── 7 · 5-fold cross-validation ──────────────────────────────────────────────

N = len(vamos_emb)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
oof_preds    = np.zeros((N, 2), dtype=np.float32)

print("5-fold cross-validation")
print(f"{'Fold':<6} {'V-CCC':>7} {'V-r':>7} {'A-CCC':>7} {'A-r':>7}")
print("-" * 40)

for fold, (tr_idx, val_idx) in enumerate(kf.split(vamos_emb)):
    model = train_va(vamos_emb[tr_idx], va[tr_idx],
                     vamos_emb[val_idx], va[val_idx], seed=fold)

    val_loader = DataLoader(VADataset(vamos_emb[val_idx], va[val_idx]),
                            batch_size=128, shuffle=False)
    preds, _, metrics = evaluate_va(model, val_loader)
    oof_preds[val_idx] = preds
    fold_metrics.append(metrics)

    print(f"  {fold+1:<4} "
          f"{metrics['valence']['ccc']:>7.4f} "
          f"{metrics['valence']['r']:>7.4f} "
          f"{metrics['arousal']['ccc']:>7.4f} "
          f"{metrics['arousal']['r']:>7.4f}")

v_ccc = np.mean([m["valence"]["ccc"] for m in fold_metrics])
a_ccc = np.mean([m["arousal"]["ccc"] for m in fold_metrics])
v_r   = np.mean([m["valence"]["r"]   for m in fold_metrics])
a_r   = np.mean([m["arousal"]["r"]   for m in fold_metrics])
print("-" * 40)
print(f"  mean  {v_ccc:>7.4f} {v_r:>7.4f} {a_ccc:>7.4f} {a_r:>7.4f}")

# ── 8 · OOF predictions plot ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, dim, i, ccc, r in [
    (axes[0], "Valence", 0, v_ccc, v_r),
    (axes[1], "Arousal", 1, a_ccc, a_r),
]:
    ax.scatter(va[:,i], oof_preds[:,i], alpha=0.3, s=12, color="steelblue")
    lo, hi = va[:,i].min(), va[:,i].max()
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2)
    ax.set_xlabel(f"True {dim} (normalised)")
    ax.set_ylabel(f"Predicted {dim}")
    ax.set_title(f"{dim} — OOF  CCC={ccc:.4f}  r={r:.4f}")
    ax.grid(alpha=0.3)
plt.suptitle("Out-of-fold predictions (5-fold CV)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/oof_predictions.png", dpi=150)
plt.show()

# ── 9 · Train final model on all 900 images ───────────────────────────────────

print("\nTraining final model on all 900 images ...")
tr_idx_final, val_idx_final = train_test_split(
    np.arange(N), test_size=0.1, random_state=42
)
final_model = train_va(
    vamos_emb[tr_idx_final], va[tr_idx_final],
    vamos_emb[val_idx_final], va[val_idx_final],
    seed=0
)
torch.save(final_model.state_dict(), f"{SAVE_DIR}/va_head_final.pt")
print("Saved: va_head_final.pt")

# ── 10 · Apply to LaMem clip_mat.npy ─────────────────────────────────────────

print("\nApplying to LaMem ...")
clip_mat = np.load(f"{FEATURES_DIR}/clip_embeddings.npy").astype(np.float32)

final_model.eval()
batch_size = 1024
va_pred_lamem = []

with torch.no_grad():
    for i in range(0, len(clip_mat), batch_size):
        batch = torch.tensor(clip_mat[i:i+batch_size]).to(DEVICE)
        va_pred_lamem.append(final_model(batch).cpu().numpy())

va_pred_lamem = np.concatenate(va_pred_lamem, axis=0)   # (N_lamem, 2)
np.save(f"{SAVE_DIR}/lamem_va_predicted.npy", va_pred_lamem)
print(f"Saved: lamem_va_predicted.npy  shape={va_pred_lamem.shape}")
print(f"Valence predicted: [{va_pred_lamem[:,0].min():.3f}, {va_pred_lamem[:,0].max():.3f}]")
print(f"Arousal predicted: [{va_pred_lamem[:,1].min():.3f}, {va_pred_lamem[:,1].max():.3f}]")

# ── 11 · Memorability prediction experiments ──────────────────────────────────

from torch.utils.data import Dataset as TorchDataset

lamem_df       = pd.read_parquet(f"{FEATURES_DIR2}/lamem_features_full.parquet")
memscore       = lamem_df["memscore"].values.astype(np.float32)
EMOTIONS_6     = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]
# emotion_scores = lamem_df[[f"emotion_bert_{e}" for e in EMOTIONS_6]].values.astype(np.float32)

all_idx    = np.arange(len(lamem_df))
strat_bins = pd.qcut(lamem_df["memscore"], q=5, labels=False, duplicates="drop")
train_idx, temp_idx = train_test_split(all_idx, test_size=0.3, random_state=42,
                                        stratify=strat_bins)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.5, random_state=42,
                                        stratify=strat_bins.iloc[temp_idx].values)


def norm_block(mat, idx):
    mu  = mat[idx].mean(0, keepdims=True)
    std = mat[idx].std(0,  keepdims=True) + 1e-8
    return ((mat - mu) / std).astype(np.float32)


clip_norm    = norm_block(clip_mat,      train_idx)
# emo_norm     = norm_block(emotion_scores, train_idx)
va_norm      = norm_block(va_pred_lamem,  train_idx)


class MemDataset(TorchDataset):
    def __init__(self, X, scores, indices):
        self.x = torch.tensor(X[indices], dtype=torch.float32)
        self.y = torch.tensor(scores[indices], dtype=torch.float32)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]


def make_loaders(X, batch_size=512):
    kw = dict(num_workers=2, pin_memory=True)
    return (
        DataLoader(MemDataset(X, memscore, train_idx), batch_size, shuffle=True,  **kw),
        DataLoader(MemDataset(X, memscore, val_idx),   batch_size, shuffle=False, **kw),
        DataLoader(MemDataset(X, memscore, test_idx),  batch_size, shuffle=False, **kw),
    )


class LinearProbe(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Linear(d, 1)
    def forward(self, x): return self.net(x).squeeze(-1)


class MLP(nn.Module):
    def __init__(self, d, hidden=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden//2, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def evaluate_mem(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(DEVICE)).cpu())
            targets.append(y)
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse     = float(np.mean((preds - targets)**2))
    r, _    = pearsonr(preds, targets)
    rho, _  = spearmanr(preds, targets)
    return mse, float(r), float(rho)


def train_mem(model, X, lr=1e-3, weight_decay=1e-4,
              max_epochs=200, patience=20, batch_size=512, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    tr, val, te = make_loaders(X, batch_size)
    model       = model.to(DEVICE)
    opt         = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion   = nn.MSELoss()
    best_mse    = float("inf")
    best_state  = None
    patience_left = patience

    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, y in tr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            criterion(model(x), y).backward()
            opt.step()

        val_mse, _, _ = evaluate_mem(model, val)
        if val_mse < best_mse - 1e-6:
            best_mse    = val_mse
            best_state  = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    test_mse, test_r, test_rho = evaluate_mem(model, te)
    return model, {"test_mse": test_mse, "test_r": test_r, "test_rho": test_rho}


EXPERIMENTS = {
    "A_vit_only":    (clip_norm,                                          768),
    # "C_vit_emo":     (np.concatenate([clip_norm, emo_norm], 1),           768 + 6),
    "M_vit_va":      (np.concatenate([clip_norm, va_norm],  1),           768 + 2),
    # "N_vit_va_emo":  (np.concatenate([clip_norm, va_norm, emo_norm], 1),  768 + 2 + 6),
    "O_va_only":     (va_norm,                                            2),
    # "P_va_emo_only": (np.concatenate([va_norm, emo_norm], 1),             2 + 6),
}

MLP_EXPERIMENTS = {
    "A_vit_only":    (clip_norm,                                          768),
    # "C_vit_emo":     (np.concatenate([clip_norm, emo_norm], 1),           768 + 6),
    "M_vit_va":      (np.concatenate([clip_norm, va_norm],  1),           768 + 2),
    # "N_vit_va_emo":  (np.concatenate([clip_norm, va_norm, emo_norm], 1),  768 + 2 + 6),
}

results_linear = {}
print("\n" + "="*65)
print("LINEAR PROBE")
print(f"{'Experiment':<22} {'MSE':>8} {'r':>8} {'ρ':>8}")
print("-" * 65)
for name, (X, dim) in EXPERIMENTS.items():
    _, hist = train_mem(LinearProbe(dim), X)
    results_linear[name] = hist
    print(f"  {name:<20} {hist['test_mse']:>8.5f} {hist['test_r']:>8.4f} {hist['test_rho']:>8.4f}")

results_mlp = {}
print("\n" + "="*65)
print("MLP")
print(f"{'Experiment':<22} {'MSE':>8} {'r':>8} {'ρ':>8}")
print("-" * 65)
for name, (X, dim) in MLP_EXPERIMENTS.items():
    _, hist = train_mem(MLP(dim), X)
    results_mlp[name] = hist
    print(f"  {name:<20} {hist['test_mse']:>8.5f} {hist['test_r']:>8.4f} {hist['test_rho']:>8.4f}")

# ── 12 · Results table ────────────────────────────────────────────────────────

baseline_r = results_linear["A_vit_only"]["test_r"]
rows = []
for name, h in results_linear.items():
    rows.append({"Experiment": name, "Model": "Linear",
                 "Test r": round(h["test_r"],4), "Test ρ": round(h["test_rho"],4),
                 "Δr vs baseline": round(h["test_r"] - baseline_r, 4)})
for name, h in results_mlp.items():
    rows.append({"Experiment": name, "Model": "MLP",
                 "Test r": round(h["test_r"],4), "Test ρ": round(h["test_rho"],4),
                 "Δr vs baseline": round(h["test_r"] - baseline_r, 4)})

res_df = pd.DataFrame(rows).sort_values(["Model","Test r"], ascending=[True,False])
print("\n" + res_df.to_string(index=False))
res_df.to_csv(f"{SAVE_DIR}/memorability_va_results.csv", index=False)

# ── 13 · Comparison bar chart ─────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

for ax, model_tag, res in [(axes[0], "Linear", results_linear),
                            (axes[1], "MLP",    results_mlp)]:
    names   = list(res.keys())
    r_vals  = [res[n]["test_r"]   for n in names]
    rho_vals= [res[n]["test_rho"] for n in names]
    x = np.arange(len(names)); w = 0.38

    bars1 = ax.bar(x - w/2, r_vals,   w, label="Pearson r",  color="steelblue",  alpha=0.85)
    bars2 = ax.bar(x + w/2, rho_vals, w, label="Spearman ρ", color="darkorange", alpha=0.85)
    ax.axhline(baseline_r, color="black", linestyle="--", linewidth=1,
               label=f"Baseline r={baseline_r:.4f}")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Correlation"); ax.set_title(f"{model_tag} — r vs ρ")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.08)

plt.suptitle("Memorability prediction — VA vs Emotion features", fontsize=13)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/memorability_comparison.png", dpi=150)
plt.show()

# ── 14 · VA feature attribution ───────────────────────────────────────────────
# How much does each predicted VA dimension contribute on its own?

va_only_model, va_only_hist = train_mem(LinearProbe(2), va_norm)
print(f"\nO_va_only  r={va_only_hist['test_r']:.4f}  ρ={va_only_hist['test_rho']:.4f}")

weights = va_only_model.net.weight.detach().cpu().numpy().flatten()
print(f"Valence weight: {weights[0]:+.4f}")
print(f"Arousal weight: {weights[1]:+.4f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, dim, i in [(axes[0], "Valence", 0), (axes[1], "Arousal", 1)]:
    _, _, te_loader = make_loaders(va_norm)
    preds, targets = [], []
    va_only_model.eval()
    with torch.no_grad():
        for x, y in te_loader:
            preds.append(va_only_model(x.to(DEVICE)).cpu().numpy())
            targets.append(y.numpy())
    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)
    r_va, _ = pearsonr(va_pred_lamem[test_idx, i], targets)

    ax.scatter(va_pred_lamem[test_idx, i], targets, alpha=0.15, s=8, color="steelblue")
    xs = np.linspace(va_pred_lamem[test_idx,i].min(), va_pred_lamem[test_idx,i].max(), 100)
    m, b = np.polyfit(va_pred_lamem[test_idx, i], targets, 1)
    ax.plot(xs, m*xs+b, color="red", linewidth=1.5)
    ax.set_xlabel(f"Predicted {dim}"); ax.set_ylabel("Memorability")
    ax.set_title(f"{dim} vs Memorability  r={r_va:.4f}")
    ax.grid(alpha=0.3)

plt.suptitle("Predicted VA vs LaMem memorability scores (test set)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/va_memorability_scatter.png", dpi=150)
plt.show()

print(f"\nAll outputs saved to: {SAVE_DIR}")
print(f"  va_head_final.pt          — trained VA regression model")
print(f"  lamem_va_predicted.npy    — VA predictions for all LaMem images (N, 2)")
print(f"  memorability_va_results.csv")
