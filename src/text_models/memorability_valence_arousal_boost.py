import os
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

try:
    import shap  # type: ignore
    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = f"{ROOT_DIR}/VAMOS_Set_1_img_info.csv"
FEATURES_DIR = f"{ROOT_DIR}/memcat_vit_results"
SAVE_DIR = f"{ROOT_DIR}/memorability_models_va_boost"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

VA_FEATURES = ["Mean_Valence", "Mean_Arousal"]
VA_LABELS = ["valence", "arousal"]
VIT_DIM = 768
VA_DIM = len(VA_FEATURES)


def norm_block(mat, idx):
    mu = mat[idx].mean(0, keepdims=True)
    std = mat[idx].std(0, keepdims=True) + 1e-8
    return ((mat - mu) / std).astype(np.float32)


class MemDataset(Dataset):
    def __init__(self, x, y, idx):
        self.x = torch.tensor(x[idx], dtype=torch.float32)
        self.y = torch.tensor(y[idx], dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class LinearProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


class FiLMProbe(nn.Module):
    """Valence/arousal vector generates (gamma, beta) over ViT features."""

    def __init__(self, vit_dim=VIT_DIM, va_dim=VA_DIM, hidden=32):
        super().__init__()
        self.vit_dim = vit_dim
        self.film = nn.Sequential(
            nn.Linear(va_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * vit_dim),
        )
        self.output = nn.Linear(vit_dim, 1)

    def forward(self, x):
        vit = x[:, : self.vit_dim]
        va = x[:, self.vit_dim :]
        params = self.film(va)
        gamma = params[:, : self.vit_dim]
        beta = params[:, self.vit_dim :]
        return self.output(gamma * vit + beta).squeeze(-1)


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(DEVICE)).cpu()
            preds.append(out)
            targets.append(yb)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse = float(np.mean((preds - targets) ** 2))
    r, _ = pearsonr(preds, targets)
    rho, _ = spearmanr(preds, targets)
    return mse, float(r), float(rho)


def make_loaders(x, y, train_idx, val_idx, test_idx, batch_size=256):
    # Safe setting for macOS.
    num_workers = 0 if os.name == "posix" and "darwin" in os.uname().sysname.lower() else 2
    kw = dict(num_workers=num_workers, pin_memory=(DEVICE == "cuda"))
    return (
        DataLoader(MemDataset(x, y, train_idx), batch_size=batch_size, shuffle=True, **kw),
        DataLoader(MemDataset(x, y, val_idx), batch_size=batch_size, shuffle=False, **kw),
        DataLoader(MemDataset(x, y, test_idx), batch_size=batch_size, shuffle=False, **kw),
    )


def train_model(model, x, y, train_idx, val_idx, test_idx, lr=1e-3, wd=1e-4, epochs=200, patience=20):
    torch.manual_seed(42)
    np.random.seed(42)
    tr, va, te = make_loaders(x, y, train_idx, val_idx, test_idx)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    best_mse = float("inf")
    best_state = None
    left = patience
    hist = {"val_r": [], "val_mse": []}

    for _ in range(epochs):
        model.train()
        for xb, yb in tr:
            opt.zero_grad()
            pred = model(xb.to(DEVICE))
            loss = nn.functional.mse_loss(pred, yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vmse, vr, _ = evaluate(model, va)
        hist["val_mse"].append(vmse)
        hist["val_r"].append(vr)
        sched.step(vmse)

        if vmse < best_mse - 1e-6:
            best_mse = vmse
            best_state = copy.deepcopy(model.state_dict())
            left = patience
        else:
            left -= 1
            if left == 0:
                break

    model.load_state_dict(best_state)
    tmse, tr, trho = evaluate(model, te)
    hist["test_mse"] = tmse
    hist["test_r"] = tr
    hist["test_rho"] = trho
    return model, hist


print("Loading VAMOS dataset...")
df = pd.read_csv(CSV_PATH)
clip_mat = np.load(f"{FEATURES_DIR}/clip_embeddings.npy").astype(np.float32)

required_cols = ["Img", "Memorability_Score"] + VA_FEATURES
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in CSV: {missing}")

if len(df) != len(clip_mat):
    raise ValueError(f"Row mismatch: CSV has {len(df)} rows, embeddings have {len(clip_mat)} rows.")

valid_mask = df[["Memorability_Score"] + VA_FEATURES].notna().all(axis=1).values
if not np.all(valid_mask):
    n_drop = int((~valid_mask).sum())
    print(f"Dropping {n_drop} rows with missing Memorability/Valence/Arousal.")
    df = df.loc[valid_mask].reset_index(drop=True)
    clip_mat = clip_mat[valid_mask]

memscore = df["Memorability_Score"].values.astype(np.float32)
va_scores = df[VA_FEATURES].values.astype(np.float32)

all_idx = np.arange(len(df))
strat_bins = pd.qcut(memscore, q=5, labels=False, duplicates="drop")
train_idx, temp_idx = train_test_split(all_idx, test_size=0.3, random_state=42, stratify=strat_bins)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=42,
    stratify=strat_bins[temp_idx],
)

clip_norm = norm_block(clip_mat, train_idx)
va_norm = norm_block(va_scores, train_idx)

results = {}


def run(tag, model, x, desc):
    print(f"\n{'='*60}\n{tag} | {desc}\ndim={x.shape[1]}")
    m, h = train_model(model, x, memscore, train_idx, val_idx, test_idx)
    print(f"test MSE={h['test_mse']:.5f}  r={h['test_r']:.4f}  rho={h['test_rho']:.4f}")
    results[tag] = {"model": m, "history": h, "desc": desc, "X": x}


# A: ViT only.
run("A_vit_only", LinearProbe(VIT_DIM), clip_mat, "ViT only baseline")

# B: ViT + raw valence/arousal.
X_b = np.concatenate([clip_mat, va_scores], axis=1)
run("B_vit_va_raw", LinearProbe(VIT_DIM + VA_DIM), X_b, "ViT + valence/arousal (raw)")

# C: ViT + normalized valence/arousal.
X_c = np.concatenate([clip_norm, va_norm], axis=1)
run("C_vit_va_norm", LinearProbe(VIT_DIM + VA_DIM), X_c, "ViT + valence/arousal (normalized)")

# D: FiLM conditioning by valence/arousal.
run("D_film_va", FiLMProbe(VIT_DIM, VA_DIM, hidden=32), X_c, "FiLM with valence/arousal conditioning")


print("\n" + "=" * 60)
print("LASSO + VA BOOST")
X_lasso = X_c.astype(np.float64)
y_train = memscore[train_idx].astype(np.float64)
y_val = memscore[val_idx].astype(np.float64)
y_test = memscore[test_idx].astype(np.float64)

lasso = LassoCV(cv=5, max_iter=8000, n_alphas=80, random_state=42)
lasso.fit(X_lasso[train_idx], y_train)
lasso_pred = lasso.predict(X_lasso[test_idx])
lasso_r, _ = pearsonr(lasso_pred, y_test)
lasso_rho, _ = spearmanr(lasso_pred, y_test)
lasso_mse = float(np.mean((lasso_pred - y_test) ** 2))
print(f"Lasso: alpha={lasso.alpha_:.6f} MSE={lasso_mse:.5f} r={lasso_r:.4f} rho={lasso_rho:.4f}")

boost_grid = [1.0, 2.0, 4.0, 8.0, 16.0]
best_boost, best_model, best_val_r = 1.0, None, -np.inf
for b in boost_grid:
    x_boost = np.concatenate([clip_norm, b * va_norm], axis=1).astype(np.float64)
    m = LassoCV(cv=5, max_iter=8000, n_alphas=80, random_state=42)
    m.fit(x_boost[train_idx], y_train)
    val_pred = m.predict(x_boost[val_idx])
    vr, _ = pearsonr(val_pred, y_val)
    if vr > best_val_r:
        best_val_r = vr
        best_boost = b
        best_model = m

X_boost_best = np.concatenate([clip_norm, best_boost * va_norm], axis=1).astype(np.float64)
boost_pred = best_model.predict(X_boost_best[test_idx])
boost_r, _ = pearsonr(boost_pred, y_test)
boost_rho, _ = spearmanr(boost_pred, y_test)
boost_mse = float(np.mean((boost_pred - y_test) ** 2))
print(
    f"VA-Boost Lasso: boost={best_boost:.1f} val_r={best_val_r:.4f} "
    f"MSE={boost_mse:.5f} r={boost_r:.4f} rho={boost_rho:.4f}"
)


if SHAP_AVAILABLE:
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS (VA-Boost Lasso)")
    bg_size = min(300, len(train_idx))
    ts_size = min(500, len(test_idx))
    bg_local = np.random.choice(len(train_idx), bg_size, replace=False)
    ts_local = np.random.choice(len(test_idx), ts_size, replace=False)
    X_bg = X_boost_best[train_idx][bg_local]
    X_eval = X_boost_best[test_idx][ts_local]

    explainer = shap.LinearExplainer(best_model, X_bg)
    sv = explainer.shap_values(X_eval)
    if isinstance(sv, list):
        sv = sv[0]

    va_mean_abs = np.abs(sv[:, VIT_DIM:]).mean(axis=0)
    vit_mean_abs = np.abs(sv[:, :VIT_DIM]).mean()
    print(f"Mean |SHAP| ViT: {vit_mean_abs:.6f}")
    print(f"Mean |SHAP| valence: {va_mean_abs[0]:.6f}")
    print(f"Mean |SHAP| arousal: {va_mean_abs[1]:.6f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(VA_LABELS, va_mean_abs, color=["#4C72B0", "#DD8452"], edgecolor="white")
    ax.axhline(vit_mean_abs, color="black", linestyle="--", linewidth=1.2, label=f"ViT mean={vit_mean_abs:.6f}")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Valence/Arousal SHAP contribution")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/shap_va_weight.png", dpi=170)
    plt.show()

    plt.figure(figsize=(8, 4))
    shap.summary_plot(
        sv[:, VIT_DIM:],
        X_eval[:, VIT_DIM:],
        feature_names=VA_LABELS,
        show=False,
    )
    plt.title("SHAP beeswarm - valence/arousal")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/shap_va_beeswarm.png", dpi=170, bbox_inches="tight")
    plt.show()
else:
    print("\nSHAP not installed; skipping SHAP plots. Install with: python3 -m pip install shap")


baseline_r = results["A_vit_only"]["history"]["test_r"]
baseline_rho = results["A_vit_only"]["history"]["test_rho"]
rows = []
for tag, v in results.items():
    h = v["history"]
    rows.append(
        {
            "Experiment": tag,
            "Description": v["desc"],
            "Test MSE": round(h["test_mse"], 5),
            "Test r": round(h["test_r"], 4),
            "Test rho": round(h["test_rho"], 4),
            "Dr": round(h["test_r"] - baseline_r, 4),
            "Drho": round(h["test_rho"] - baseline_rho, 4),
        }
    )

rows.append(
    {
        "Experiment": "Lasso",
        "Description": "LassoCV on ViT+VA norm",
        "Test MSE": round(lasso_mse, 5),
        "Test r": round(lasso_r, 4),
        "Test rho": round(lasso_rho, 4),
        "Dr": round(lasso_r - baseline_r, 4),
        "Drho": round(lasso_rho - baseline_rho, 4),
    }
)
rows.append(
    {
        "Experiment": "Lasso_VA_Boost",
        "Description": f"LassoCV with VA scaling (x{best_boost:.1f})",
        "Test MSE": round(boost_mse, 5),
        "Test r": round(boost_r, 4),
        "Test rho": round(boost_rho, 4),
        "Dr": round(boost_r - baseline_r, 4),
        "Drho": round(boost_rho - baseline_rho, 4),
    }
)

table = pd.DataFrame(rows).sort_values("Test r", ascending=False)
print("\n" + "=" * 60)
print("FINAL RESULTS")
print(table.to_string(index=False))
table.to_csv(f"{SAVE_DIR}/results_va_boost.csv", index=False)

fig, ax = plt.subplots(figsize=(12, 5))
for tag, color in [
    ("A_vit_only", "steelblue"),
    ("B_vit_va_raw", "orange"),
    ("C_vit_va_norm", "green"),
    ("D_film_va", "purple"),
]:
    if tag in results:
        ax.plot(results[tag]["history"]["val_r"], label=tag, color=color, linewidth=1.6)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Pearson r")
ax.set_title("Validation Pearson r - VAMOS (valence/arousal)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/learning_curves_va.png", dpi=150)
plt.show()

print("\nAll outputs saved to:", SAVE_DIR)
