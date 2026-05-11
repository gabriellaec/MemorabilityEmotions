

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LassoCV, Lasso, ElasticNetCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import copy
import json
import os
from sklearn.model_selection import train_test_split

# SHAP is optional: the script can run training/evaluation without it.
try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore

# from google.colab import drive
# drive.mount("/content/drive", force_remount=False)


# Resolve project root from this file location for portability.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

FEATURES_DIR = f"{ROOT_DIR}/lamem_features"
FEATURES_DIR3 = f"{ROOT_DIR}/lamem_features_v3"
SAVE_DIR     = f"{ROOT_DIR}/memorability_models_boost"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

EMOTIONS = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]
VIT_DIM  = 768
EMO_DIM  = len(EMOTIONS)

# ── 1 · Data ──────────────────────────────────────────────────────────────────

df       = pd.read_parquet(f"{FEATURES_DIR3}/lamem_features_emotions_full.parquet")
clip_mat = np.load(f"{FEATURES_DIR}/clip_embeddings.npy").astype(np.float32)

def _basename_key(series):
    return series.astype(str).str.split("/").str[-1]

full_meta_path = f"{FEATURES_DIR}/lamem_features_full.parquet"
if os.path.exists(full_meta_path):
    full_df = pd.read_parquet(full_meta_path)
    full_keys = _basename_key(full_df["name"])
    if full_keys.duplicated().any():
        raise ValueError("Duplicate image names found in lamem_features_full.parquet; cannot align embeddings safely.")

    df_key_source = "name" if "name" in df.columns else ("image" if "image" in df.columns else None)
    if df_key_source is None:
        raise KeyError("Neither 'name' nor 'image' column found in emotion dataframe for embedding alignment.")

    df_keys = _basename_key(df[df_key_source])
    key_to_idx = pd.Series(np.arange(len(full_df)), index=full_keys)
    clip_indices = key_to_idx.reindex(df_keys)

    if clip_indices.isna().any():
        n_missing = int(clip_indices.isna().sum())
        raise KeyError(f"Could not find {n_missing} emotion rows in full metadata; embedding alignment failed.")

    clip_mat = clip_mat[clip_indices.astype(np.int64).to_numpy()]
    print(f"Aligned embeddings by '{df_key_source}' against lamem_features_full.parquet.")
else:
    # Fallback only when full metadata is unavailable.
    if len(df) != len(clip_mat):
        n = min(len(df), len(clip_mat))
        print(
            f"Row mismatch detected (df={len(df)}, clip={len(clip_mat)}). "
            f"Truncating both to {n} rows because full metadata was not found."
        )
        df = df.iloc[:n].reset_index(drop=True)
        clip_mat = clip_mat[:n]

memscore      = df["memscore"].values.astype(np.float32)

def _pick_emotion_columns(frame, emotions):
    candidates = [
        [f"emotion_bert_{e}" for e in emotions],
        [f"emotion_roberta_softmax_{e}" for e in emotions],
        [f"emotion_go_softmax_{e}" for e in emotions],
        [f"emotion_{e}" for e in emotions],
    ]
    for cols in candidates:
        if all(c in frame.columns for c in cols):
            return cols
    raise KeyError(
        f"No supported emotion columns found for emotions {emotions}. "
        "Expected one of: emotion_bert_*, emotion_roberta_softmax_*, emotion_go_softmax_*, emotion_*."
    )

emotion_cols = _pick_emotion_columns(df, EMOTIONS)
emotion_scores = df[emotion_cols].values.astype(np.float32)

entropy_col_candidates = [
    "emotion_bert_entropy",
    "emotion_roberta_entropy",
    "emotion_go_entropy",
]
entropy_col = next((c for c in entropy_col_candidates if c in df.columns), None)
if entropy_col is not None:
    entropy_feat = df[[entropy_col]].values.astype(np.float32)
else:
    # Compute entropy from emotion probabilities when dataset does not provide it.
    probs = np.clip(emotion_scores, 1e-8, 1.0)
    entropy_feat = (-np.sum(probs * np.log(probs), axis=1, keepdims=True)).astype(np.float32)

print(f"Using emotion columns: {emotion_cols}")
emotion_labels = np.argmax(emotion_scores, axis=1).astype(np.int64)

all_idx    = np.arange(len(df))
strat_bins = pd.qcut(df["memscore"], q=5, labels=False, duplicates="drop")
train_idx, temp_idx = train_test_split(all_idx, test_size=0.3, random_state=42, stratify=strat_bins)
val_idx, test_idx   = train_test_split(temp_idx, test_size=0.5, random_state=42,
                                        stratify=strat_bins.iloc[temp_idx].values)

# ── 2 · Block-wise normalization (fit on train only) ─────────────────────────

def norm_block(mat, idx):
    mu  = mat[idx].mean(0, keepdims=True)
    std = mat[idx].std(0,  keepdims=True) + 1e-8
    return ((mat - mu) / std).astype(np.float32)

clip_norm = norm_block(clip_mat,      train_idx)
emo_norm  = norm_block(emotion_scores, train_idx)
ent_norm  = norm_block(entropy_feat,   train_idx)

# ── 3 · Dataset & loaders ─────────────────────────────────────────────────────

class MemDataset(Dataset):
    def __init__(self, X, scores, indices, emo_labels=None):
        self.x   = torch.tensor(X[indices], dtype=torch.float32)
        self.y   = torch.tensor(scores[indices], dtype=torch.float32)
        self.emo = torch.tensor(emo_labels[indices], dtype=torch.long) if emo_labels is not None else None

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        if self.emo is not None:
            return self.x[i], self.y[i], self.emo[i]
        return self.x[i], self.y[i]


def make_loaders(X, batch_size=512, emo_labels=None):
    # macOS + top-level scripts can fail with multi-worker DataLoader spawn.
    # Use single-process loading by default on Darwin for stability.
    num_workers = 0 if os.name == "posix" and "darwin" in os.uname().sysname.lower() else 2
    kw = dict(num_workers=num_workers, pin_memory=(DEVICE == "cuda"))
    return (
        DataLoader(MemDataset(X, memscore, train_idx, emo_labels), batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(MemDataset(X, memscore, val_idx,   emo_labels), batch_size=batch_size, shuffle=False, **kw),
        DataLoader(MemDataset(X, memscore, test_idx,  emo_labels), batch_size=batch_size, shuffle=False, **kw),
    )

# ── 4 · Evaluation ────────────────────────────────────────────────────────────

def evaluate(model, loader, has_emo_label=False):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[0].to(DEVICE), batch[1]
            out = model(x)
            p   = out[0] if isinstance(out, tuple) else out
            preds.append(p.cpu()); targets.append(y)
    preds   = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse     = float(np.mean((preds - targets) ** 2))
    r, _    = pearsonr(preds, targets)
    rho, _  = spearmanr(preds, targets)
    return mse, float(r), float(rho)

# ── 5 · Models ────────────────────────────────────────────────────────────────

class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    def forward(self, x):
        return self.linear(x).squeeze(-1)


class FiLMProbe(nn.Module):
    """Emotion vector generates (γ, β) that scale and shift ViT features."""
    def __init__(self, vit_dim=VIT_DIM, emo_dim=EMO_DIM, hidden=64):
        super().__init__()
        self.vit_dim = vit_dim
        self.film    = nn.Sequential(
            nn.Linear(emo_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * vit_dim),
        )
        self.output  = nn.Linear(vit_dim, 1)

    def forward(self, x):
        vit    = x[:, :self.vit_dim]
        emo    = x[:, self.vit_dim:]
        params = self.film(emo)
        gamma  = params[:, :self.vit_dim]
        beta   = params[:, self.vit_dim:]
        return self.output(gamma * vit + beta).squeeze(-1)


class RankMLP(nn.Module):
    """MLP trained with a Spearman-surrogate rank loss."""
    def __init__(self, input_dim, hidden=256, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class AuxMLP(nn.Module):
    """Shared trunk → mem head + emotion classification head."""
    def __init__(self, input_dim, n_emotions=EMO_DIM, hidden=256, dropout=0.15):
        super().__init__()
        self.trunk    = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
        )
        self.mem_head = nn.Linear(hidden // 2, 1)
        self.emo_head = nn.Linear(hidden // 2, n_emotions)

    def forward(self, x):
        h = self.trunk(x)
        return self.mem_head(h).squeeze(-1), self.emo_head(h)

# ── 6 · Loss functions ────────────────────────────────────────────────────────

def soft_rank(x, tau=0.1):
    diff = (x.unsqueeze(1) - x.unsqueeze(0)) / tau
    return torch.sigmoid(diff).sum(dim=1)


def spearman_loss(preds, targets):
    p_ranks = soft_rank(preds)
    t_ranks = soft_rank(targets)
    p_ranks = p_ranks - p_ranks.mean()
    t_ranks = t_ranks - t_ranks.mean()
    cos = (p_ranks * t_ranks).sum() / (p_ranks.norm() * t_ranks.norm() + 1e-8)
    return 1.0 - cos

# ── 7 · Generic trainer ───────────────────────────────────────────────────────

def train_model(
    model, X, loss_fn,
    lr=1e-3, weight_decay=1e-4,
    max_epochs=200, patience=20, batch_size=512,
    seed=42, emo_labels=None, aux_alpha=0.3,
):
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader, val_loader, test_loader = make_loaders(X, batch_size, emo_labels)
    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    history       = {"train_loss": [], "val_mse": [], "val_r": [], "val_rho": []}
    best_val_mse  = float("inf")
    best_state    = None
    patience_left = patience
    has_emo       = emo_labels is not None
    ce_loss       = nn.CrossEntropyLoss()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            opt.zero_grad()
            out  = model(x)

            if isinstance(out, tuple):
                mem_pred, emo_pred = out
                emo_true = batch[2].to(DEVICE)
                loss = (1 - aux_alpha) * loss_fn(mem_pred, y) + aux_alpha * ce_loss(emo_pred, emo_true)
            else:
                loss = loss_fn(out, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item() * len(y)

        val_mse, val_r, val_rho = evaluate(model, val_loader, has_emo)
        history["train_loss"].append(epoch_loss / len(train_loader.dataset))
        history["val_mse"].append(val_mse)
        history["val_r"].append(val_r)
        history["val_rho"].append(val_rho)
        sched.step(val_mse)

        if val_mse < best_val_mse - 1e-6:
            best_val_mse  = val_mse
            best_state    = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    test_mse, test_r, test_rho = evaluate(model, test_loader, has_emo)
    history["test_mse"] = test_mse
    history["test_r"]   = test_r
    history["test_rho"] = test_rho
    return model, history

# ── 8 · Experiment registry ───────────────────────────────────────────────────

results = {}

def run(tag, model, X, loss_fn, desc, emo_labels=None, aux_alpha=0.3, batch_size=512):
    print(f"\n{'='*60}\n{tag}  |  {desc}\ndim={X.shape[1]}")
    m, h = train_model(model, X, loss_fn, batch_size=batch_size,
                       emo_labels=emo_labels, aux_alpha=aux_alpha)
    print(f"test MSE={h['test_mse']:.5f}  r={h['test_r']:.4f}  ρ={h['test_rho']:.4f}")
    results[tag] = {"model": m, "history": h, "desc": desc, "X": X}
    return m, h

mse_loss = nn.MSELoss()

# ── A: baseline (raw concat, MSE) ─────────────────────────────────────────────
run("A_vit_only",
    LinearProbe(VIT_DIM),
    clip_mat, mse_loss, "ViT only — baseline")

# ── C: raw concat + emotion (MSE) ─────────────────────────────────────────────
X_c = np.concatenate([clip_mat, emotion_scores], axis=1)
run("C_vit_emo",
    LinearProbe(VIT_DIM + EMO_DIM),
    X_c, mse_loss, "ViT + emotion (raw, MSE)")

# ── G: normalised blocks (MSE) ────────────────────────────────────────────────
X_g = np.concatenate([clip_norm, emo_norm], axis=1)
run("G_vit_emo_norm",
    LinearProbe(VIT_DIM + EMO_DIM),
    X_g, mse_loss, "ViT + emotion — block-normalised")

# ── H: FiLM conditioning (MSE) ────────────────────────────────────────────────
X_h = np.concatenate([clip_norm, emo_norm], axis=1)
run("H_film",
    FiLMProbe(VIT_DIM, EMO_DIM, hidden=64),
    X_h, mse_loss, "FiLM: emotion modulates ViT features")

# ── I: MLP + Spearman rank loss ───────────────────────────────────────────────
X_i = np.concatenate([clip_norm, emo_norm], axis=1)
run("I_rank_loss",
    RankMLP(VIT_DIM + EMO_DIM, hidden=256),
    X_i, spearman_loss, "MLP — Spearman rank loss", batch_size=256)

# ── J: MLP + auxiliary emotion classification head ────────────────────────────
X_j = np.concatenate([clip_norm, emo_norm], axis=1)
run("J_aux_emo",
    AuxMLP(VIT_DIM + EMO_DIM, n_emotions=EMO_DIM, hidden=256),
    X_j, mse_loss, "MLP — aux emotion head (α=0.3)",
    emo_labels=emotion_labels, aux_alpha=0.3)

# ── K: FiLM + rank loss (combined best ideas) ─────────────────────────────────
run("K_film_rank",
    FiLMProbe(VIT_DIM, EMO_DIM, hidden=64),
    X_h, spearman_loss, "FiLM + Spearman rank loss")

# ── L: aux head + rank loss ───────────────────────────────────────────────────
run("L_aux_rank",
    AuxMLP(VIT_DIM + EMO_DIM, n_emotions=EMO_DIM, hidden=256),
    X_j, spearman_loss, "MLP — aux emotion head + rank loss",
    emo_labels=emotion_labels, aux_alpha=0.3, batch_size=256)

# ── 9 · Lasso baseline ────────────────────────────────────────────────────────

print("\n" + "="*60)
print("LASSO BASELINE")

X_lasso = np.concatenate([clip_norm, emo_norm], axis=1).astype(np.float64)
y_train  = memscore[train_idx].astype(np.float64)
y_val    = memscore[val_idx].astype(np.float64)
y_test   = memscore[test_idx].astype(np.float64)

lasso_cv = LassoCV(cv=5, max_iter=5000, n_alphas=50, random_state=42)
lasso_cv.fit(X_lasso[train_idx], y_train)

lasso_pred = lasso_cv.predict(X_lasso[test_idx])
lasso_r, _ = pearsonr(lasso_pred, y_test)
lasso_rho, _ = spearmanr(lasso_pred, y_test)
lasso_mse   = float(np.mean((lasso_pred - y_test) ** 2))
print(f"alpha={lasso_cv.alpha_:.6f}  MSE={lasso_mse:.5f}  r={lasso_r:.4f}  ρ={lasso_rho:.4f}")

vit_weights = lasso_cv.coef_[:VIT_DIM]
emo_weights = lasso_cv.coef_[VIT_DIM:]

print(f"\nLasso weight statistics:")
print(f"  ViT    — nonzero: {(vit_weights != 0).sum()}/{VIT_DIM}  |  mean |w|: {np.abs(vit_weights).mean():.6f}")
print(f"  Emotion— nonzero: {(emo_weights != 0).sum()}/{EMO_DIM}   |  mean |w|: {np.abs(emo_weights).mean():.6f}")
print(f"\nEmotion weights:")
for e, w in zip(EMOTIONS, emo_weights):
    print(f"  {e:<12}  w={w:+.6f}  {'*' if w != 0 else '(zeroed)'}")

lasso_path_alphas = np.logspace(-4, 0, 50)
emo_coefs_path = np.zeros((len(lasso_path_alphas), EMO_DIM))
for i, alpha in enumerate(lasso_path_alphas):
    l = Lasso(alpha=alpha, max_iter=5000).fit(X_lasso[train_idx], y_train)
    emo_coefs_path[i] = l.coef_[VIT_DIM:]

fig, ax = plt.subplots(figsize=(10, 5))
for j, emo in enumerate(EMOTIONS):
    ax.plot(np.log10(lasso_path_alphas), emo_coefs_path[:, j], label=emo, linewidth=1.8)
ax.axvline(np.log10(lasso_cv.alpha_), color="black", linestyle="--", linewidth=1, label=f"CV alpha")
ax.set_xlabel("log₁₀(α)")
ax.set_ylabel("Lasso coefficient")
ax.set_title("Regularisation path — emotion features (Lasso)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/lasso_emotion_path.png", dpi=150)
plt.show()

# ── 9b · ElasticNet baseline ──────────────────────────────────────────────────
print("\n" + "="*60)
print("ELASTICNET BASELINE")

elastic_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    cv=5,
    max_iter=20000,
    random_state=42,
)
elastic_cv.fit(X_lasso[train_idx], y_train)
elastic_pred = elastic_cv.predict(X_lasso[test_idx])
elastic_r, _ = pearsonr(elastic_pred, y_test)
elastic_rho, _ = spearmanr(elastic_pred, y_test)
elastic_mse = float(np.mean((elastic_pred - y_test) ** 2))
print(
    f"alpha={elastic_cv.alpha_:.6f}  l1_ratio={elastic_cv.l1_ratio_:.2f}  "
    f"MSE={elastic_mse:.5f}  r={elastic_r:.4f}  ρ={elastic_rho:.4f}"
)

# ── 9c · PCA + Ridge baseline ─────────────────────────────────────────────────
print("\n" + "="*60)
print("PCA + RIDGE BASELINE")

n_components = min(256, VIT_DIM, len(train_idx) - 1)
pca = PCA(n_components=n_components, random_state=42)
vit_train_pca = pca.fit_transform(clip_norm[train_idx].astype(np.float64))
vit_test_pca = pca.transform(clip_norm[test_idx].astype(np.float64))
X_train_pca_ridge = np.concatenate([vit_train_pca, emo_norm[train_idx].astype(np.float64)], axis=1)
X_test_pca_ridge = np.concatenate([vit_test_pca, emo_norm[test_idx].astype(np.float64)], axis=1)
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 25), cv=5)
ridge_cv.fit(X_train_pca_ridge, y_train)
ridge_pred = ridge_cv.predict(X_test_pca_ridge)
ridge_r, _ = pearsonr(ridge_pred, y_test)
ridge_rho, _ = spearmanr(ridge_pred, y_test)
ridge_mse = float(np.mean((ridge_pred - y_test) ** 2))
print(
    f"alpha={ridge_cv.alpha_:.6f}  pca_dim={n_components}  "
    f"MSE={ridge_mse:.5f}  r={ridge_r:.4f}  ρ={ridge_rho:.4f}"
)

# ── 9d · Residual learning (ViT then emotion correction) ──────────────────────
print("\n" + "="*60)
print("RESIDUAL BASELINE (ViT + emotion correction)")

vit_ridge = RidgeCV(alphas=np.logspace(-4, 4, 25), cv=5)
vit_ridge.fit(clip_norm[train_idx].astype(np.float64), y_train)
train_vit_pred = vit_ridge.predict(clip_norm[train_idx].astype(np.float64))
test_vit_pred = vit_ridge.predict(clip_norm[test_idx].astype(np.float64))

residual_train = y_train - train_vit_pred
emo_residual_model = RidgeCV(alphas=np.logspace(-4, 4, 25), cv=5)
emo_residual_model.fit(emo_norm[train_idx].astype(np.float64), residual_train)
residual_test_pred = emo_residual_model.predict(emo_norm[test_idx].astype(np.float64))
residual_pred = test_vit_pred + residual_test_pred

residual_r, _ = pearsonr(residual_pred, y_test)
residual_rho, _ = spearmanr(residual_pred, y_test)
residual_mse = float(np.mean((residual_pred - y_test) ** 2))
print(
    f"vit_alpha={vit_ridge.alpha_:.6f}  emo_alpha={emo_residual_model.alpha_:.6f}  "
    f"MSE={residual_mse:.5f}  r={residual_r:.4f}  ρ={residual_rho:.4f}"
)

# ── 9e · Emotion-boosted linear baseline ──────────────────────────────────────
print("\n" + "="*60)
print("EMOTION-BOOSTED LASSO BASELINE")

boost_grid = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0]
# Trade-off knob: higher value explicitly prioritizes emotional contribution.
EMOTION_PRIORITY = 0.10

def linear_emotion_share(model, X):
    """Fraction of absolute linear contribution coming from emotion features."""
    coef = model.coef_
    vit_contrib = np.abs(X[:, :VIT_DIM] * coef[:VIT_DIM]).sum(axis=1)
    emo_contrib = np.abs(X[:, VIT_DIM:] * coef[VIT_DIM:]).sum(axis=1)
    return float(np.mean(emo_contrib / (vit_contrib + emo_contrib + 1e-12)))

best_boost = None
best_boost_val_r = -np.inf
best_boost_val_share = -np.inf
best_boost_score = -np.inf
boost_model = None

for b in boost_grid:
    X_boost = np.concatenate([clip_norm, b * emo_norm], axis=1).astype(np.float64)
    model_b = LassoCV(cv=5, max_iter=5000, n_alphas=50, random_state=42)
    model_b.fit(X_boost[train_idx], y_train)
    val_pred_b = model_b.predict(X_boost[val_idx])
    val_r_b, _ = pearsonr(val_pred_b, y_val)
    val_share_b = linear_emotion_share(model_b, X_boost[val_idx])
    score_b = val_r_b + EMOTION_PRIORITY * val_share_b
    if score_b > best_boost_score:
        best_boost_score = score_b
        best_boost_val_r = val_r_b
        best_boost_val_share = val_share_b
        best_boost = b
        boost_model = model_b

X_boost_best = np.concatenate([clip_norm, best_boost * emo_norm], axis=1).astype(np.float64)
boost_pred = boost_model.predict(X_boost_best[test_idx])
boost_r, _ = pearsonr(boost_pred, y_test)
boost_rho, _ = spearmanr(boost_pred, y_test)
boost_mse = float(np.mean((boost_pred - y_test) ** 2))
boost_test_share = linear_emotion_share(boost_model, X_boost_best[test_idx])
print(
    f"best_boost={best_boost:.1f}  val_r={best_boost_val_r:.4f}  "
    f"val_emotion_share={best_boost_val_share:.3f}  test_emotion_share={boost_test_share:.3f}  "
    f"MSE={boost_mse:.5f}  r={boost_r:.4f}  ρ={boost_rho:.4f}"
)

# ── 9f · Residual with calibrated emotion gain ────────────────────────────────
print("\n" + "="*60)
print("RESIDUAL + EMOTION GAIN CALIBRATION")

val_vit_pred = vit_ridge.predict(clip_norm[val_idx].astype(np.float64))
val_residual_pred = emo_residual_model.predict(emo_norm[val_idx].astype(np.float64))
test_residual_pred = emo_residual_model.predict(emo_norm[test_idx].astype(np.float64))

gamma_grid = np.linspace(0.0, 3.0, 31)
best_gamma = 1.0
best_gamma_val_r = -np.inf
for g in gamma_grid:
    val_pred_g = val_vit_pred + g * val_residual_pred
    r_g, _ = pearsonr(val_pred_g, y_val)
    if r_g > best_gamma_val_r:
        best_gamma_val_r = r_g
        best_gamma = float(g)

residual_gain_pred = test_vit_pred + best_gamma * test_residual_pred
residual_gain_r, _ = pearsonr(residual_gain_pred, y_test)
residual_gain_rho, _ = spearmanr(residual_gain_pred, y_test)
residual_gain_mse = float(np.mean((residual_gain_pred - y_test) ** 2))
print(
    f"best_gamma={best_gamma:.2f}  val_r={best_gamma_val_r:.4f}  "
    f"MSE={residual_gain_mse:.5f}  r={residual_gain_r:.4f}  ρ={residual_gain_rho:.4f}"
)

# ── 10 · SHAP analysis ────────────────────────────────────────────────────────
if SHAP_AVAILABLE:
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    # Use the best linear model focused on emotion contribution interpretability.
    # This avoids fragile deep-model explainers and gives directly comparable feature attributions.
    X_train_shap = X_boost_best[train_idx]
    X_test_shap = X_boost_best[test_idx]
    bg_size = min(500, len(X_train_shap))
    test_size = min(1000, len(X_test_shap))
    bg_idx = np.random.choice(len(X_train_shap), bg_size, replace=False)
    ts_idx = np.random.choice(len(X_test_shap), test_size, replace=False)
    X_bg = X_train_shap[bg_idx]
    X_eval = X_test_shap[ts_idx]

    feature_names = [f"vit_{i}" for i in range(VIT_DIM)] + [f"emo_{e}" for e in EMOTIONS]
    explainer = shap.LinearExplainer(boost_model, X_bg)
    shap_values = explainer.shap_values(X_eval)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    vit_mean_abs = np.abs(shap_values[:, :VIT_DIM]).mean()
    emo_mean_abs = np.abs(shap_values[:, VIT_DIM:]).mean()
    emo_importance = np.abs(shap_values[:, VIT_DIM:]).mean(axis=0)
    print(f"Model explained: EmotionBoost_Lasso (boost={best_boost:.1f})")
    print(f"  Mean |SHAP| ViT     : {vit_mean_abs:.6f}")
    print(f"  Mean |SHAP| Emotions: {emo_mean_abs:.6f}  (ratio: {emo_mean_abs/(vit_mean_abs + 1e-12):.2f}x)")
    for e, s in zip(EMOTIONS, emo_importance):
        print(f"    {e:<12}  {s:.6f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, EMO_DIM))
    ax.bar(EMOTIONS, emo_importance, color=colors, edgecolor="white")
    ax.axhline(vit_mean_abs, color="black", linestyle="--", linewidth=1.2,
               label=f"Mean |SHAP| ViT dims = {vit_mean_abs:.6f}")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Emotion contribution (SHAP) — EmotionBoost_Lasso")
    ax.set_xticks(np.arange(EMO_DIM))
    ax.set_xticklabels(EMOTIONS, rotation=25, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/shap_emotion_weight_linear.png", dpi=170)
    plt.show()

    plt.figure(figsize=(10, 5))
    shap.summary_plot(
        shap_values[:, VIT_DIM:],
        X_eval[:, VIT_DIM:],
        feature_names=EMOTIONS,
        show=False,
    )
    plt.title("SHAP beeswarm — emotion features (EmotionBoost_Lasso)")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/shap_emotion_beeswarm_linear.png", dpi=170, bbox_inches="tight")
    plt.show()
else:
    print("\n" + "="*60)
    print("SHAP ANALYSIS")
    print("`shap` is not installed; skipping SHAP plots/attributions. Install with: python3 -m pip install shap")

# ── 11 · Results table ────────────────────────────────────────────────────────

baseline_r   = results["A_vit_only"]["history"]["test_r"]
baseline_rho = results["A_vit_only"]["history"]["test_rho"]
baseline_mse = results["A_vit_only"]["history"]["test_mse"]

rows = []
for tag, v in results.items():
    h = v["history"]
    rows.append({
        "Experiment": tag,
        "Description": v["desc"],
        "Test MSE": round(h["test_mse"], 5),
        "Test r":   round(h["test_r"],   4),
        "Test ρ":   round(h["test_rho"],  4),
        "Δr":       round(h["test_r"]   - baseline_r,   4),
        "Δρ":       round(h["test_rho"]  - baseline_rho, 4),
    })

rows.append({
    "Experiment": "Lasso_baseline",
    "Description": "Lasso CV (sklearn)",
    "Test MSE": round(lasso_mse, 5),
    "Test r":   round(lasso_r,   4),
    "Test ρ":   round(lasso_rho, 4),
    "Δr":       round(lasso_r   - baseline_r,   4),
    "Δρ":       round(lasso_rho - baseline_rho, 4),
})

rows.append({
    "Experiment": "ElasticNet_baseline",
    "Description": "ElasticNet CV (sklearn)",
    "Test MSE": round(elastic_mse, 5),
    "Test r":   round(elastic_r, 4),
    "Test ρ":   round(elastic_rho, 4),
    "Δr":       round(elastic_r - baseline_r, 4),
    "Δρ":       round(elastic_rho - baseline_rho, 4),
})

rows.append({
    "Experiment": "PCA_Ridge_baseline",
    "Description": "PCA(clip) + Ridge CV",
    "Test MSE": round(ridge_mse, 5),
    "Test r":   round(ridge_r, 4),
    "Test ρ":   round(ridge_rho, 4),
    "Δr":       round(ridge_r - baseline_r, 4),
    "Δρ":       round(ridge_rho - baseline_rho, 4),
})

rows.append({
    "Experiment": "Residual_Ridge",
    "Description": "Ridge(clip) + Ridge(emotion residual)",
    "Test MSE": round(residual_mse, 5),
    "Test r":   round(residual_r, 4),
    "Test ρ":   round(residual_rho, 4),
    "Δr":       round(residual_r - baseline_r, 4),
    "Δρ":       round(residual_rho - baseline_rho, 4),
})

rows.append({
    "Experiment": "EmotionBoost_Lasso",
    "Description": f"Lasso CV with emotion scaling (x{best_boost:.1f})",
    "Test MSE": round(boost_mse, 5),
    "Test r":   round(boost_r, 4),
    "Test ρ":   round(boost_rho, 4),
    "Δr":       round(boost_r - baseline_r, 4),
    "Δρ":       round(boost_rho - baseline_rho, 4),
})

rows.append({
    "Experiment": "Residual_Ridge_Gamma",
    "Description": f"Residual Ridge + calibrated gamma ({best_gamma:.2f})",
    "Test MSE": round(residual_gain_mse, 5),
    "Test r":   round(residual_gain_r, 4),
    "Test ρ":   round(residual_gain_rho, 4),
    "Δr":       round(residual_gain_r - baseline_r, 4),
    "Δρ":       round(residual_gain_rho - baseline_rho, 4),
})

table = pd.DataFrame(rows).sort_values("Test r", ascending=False)
print("\n" + "="*60)
print("FINAL RESULTS")
print(table.to_string(index=False))
table.to_csv(f"{SAVE_DIR}/results_boost.csv", index=False)

# ── 12 · Comparison plots ─────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

tags  = [r["Experiment"] for _, r in enumerate(rows)]
r_vals = [r["Test r"] for r in rows]
rho_vals = [r["Test ρ"] for r in rows]

colors = ["#2ecc71" if v > baseline_r else "#e74c3c" for v in r_vals]
colors[tags.index("A_vit_only")] = "#888888"

for ax, vals, metric, bl in zip(axes, [r_vals, rho_vals], ["Pearson r", "Spearman ρ"],
                                  [baseline_r, baseline_rho]):
    bars = ax.bar(tags, vals, color=colors, edgecolor="white", alpha=0.85)
    ax.axhline(bl, color="black", linestyle="--", linewidth=1.2, label=f"Baseline = {bl:.4f}")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                f"{val:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_title(f"Test {metric} by experiment")
    ax.set_xticklabels(tags, rotation=40, ha="right", fontsize=8)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

plt.suptitle("Emotion boost experiments — test performance", fontsize=12)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/results_comparison.png", dpi=150)
plt.show()

n_epochs_plot = min(len(results["H_film"]["history"]["val_r"]),
                    len(results["J_aux_emo"]["history"]["val_r"]),
                    len(results["C_vit_emo"]["history"]["val_r"]))

fig, ax = plt.subplots(figsize=(12, 5))
for tag, color in [("C_vit_emo","steelblue"), ("G_vit_emo_norm","orange"),
                   ("H_film","green"), ("I_rank_loss","red"),
                   ("J_aux_emo","purple"), ("K_film_rank","brown"), ("L_aux_rank","pink")]:
    if tag in results:
        h = results[tag]["history"]
        ax.plot(h["val_r"], label=tag, color=color, linewidth=1.4)
ax.set_xlabel("Epoch"); ax.set_ylabel("Val Pearson r")
ax.set_title("Validation Pearson r — learning curves")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/learning_curves.png", dpi=150)
plt.show()

print("\nAll outputs saved to:", SAVE_DIR)
