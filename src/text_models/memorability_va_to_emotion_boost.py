import os
import copy
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    SHAP_AVAILABLE = False
    shap = None  # type: ignore


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = f"{ROOT_DIR}/VAMOS_Set_1_img_info.csv"
FEATURES_DIR = f"{ROOT_DIR}/memcat_vit_results"
SAVE_DIR = f"{ROOT_DIR}/memorability_models_va_to_emotions"
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

VIT_DIM = 768
VA_COLS = ["Mean_Valence", "Mean_Arousal"]
EMOTIONS = ["joy", "sadness", "fear", "anger", "disgust", "surprise"]

HARTMANN_VA_1_9 = {
    "joy": {"valence": (7, 9), "arousal": (6, 9)},
    "sadness": {"valence": (1, 3), "arousal": (2, 5)},
    "fear": {"valence": (1, 3), "arousal": (7, 9)},
    "anger": {"valence": (1, 3), "arousal": (6, 9)},
    "disgust": {"valence": (1, 4), "arousal": (4, 7)},
    "surprise": {"valence": (4, 7), "arousal": (7, 9)},
}


def norm_block(mat, idx):
    mu = mat[idx].mean(0, keepdims=True)
    std = mat[idx].std(0, keepdims=True) + 1e-8
    return ((mat - mu) / std).astype(np.float32)


def distance_to_interval(x, lo, hi):
    if lo <= x <= hi:
        return 0.0
    return min(abs(x - lo), abs(x - hi))


def membership_score(x, lo, hi):
    """Soft range membership centered on interval; values in [0, 1]."""
    span = max(hi - lo, 1e-6)
    d = distance_to_interval(x, lo, hi)
    return float(np.exp(-0.5 * (d / span) ** 2))


def va_to_emotion_scores(valence, arousal):
    raw_scores = []
    for emo in EMOTIONS:
        v_lo, v_hi = HARTMANN_VA_1_9[emo]["valence"]
        a_lo, a_hi = HARTMANN_VA_1_9[emo]["arousal"]
        sv = membership_score(valence, v_lo, v_hi)
        sa = membership_score(arousal, a_lo, a_hi)
        raw_scores.append(sv * sa)
    raw = np.asarray(raw_scores, dtype=np.float32)
    denom = float(raw.sum())
    if denom < 1e-12:
        return np.full(len(EMOTIONS), 1.0 / len(EMOTIONS), dtype=np.float32)
    return (raw / denom).astype(np.float32)


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


def make_loaders(x, y, train_idx, val_idx, test_idx, batch_size=256):
    num_workers = 0 if os.name == "posix" and "darwin" in os.uname().sysname.lower() else 2
    kw = dict(num_workers=num_workers, pin_memory=(DEVICE == "cuda"))
    return (
        DataLoader(MemDataset(x, y, train_idx), batch_size=batch_size, shuffle=True, **kw),
        DataLoader(MemDataset(x, y, val_idx), batch_size=batch_size, shuffle=False, **kw),
        DataLoader(MemDataset(x, y, test_idx), batch_size=batch_size, shuffle=False, **kw),
    )


def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(DEVICE)).cpu()
            preds.append(pred)
            targets.append(yb)
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    mse = float(np.mean((preds - targets) ** 2))
    r, _ = pearsonr(preds, targets)
    rho, _ = spearmanr(preds, targets)
    return mse, float(r), float(rho)


def train_model(model, x, y, train_idx, val_idx, test_idx, epochs=200, patience=20):
    torch.manual_seed(42)
    np.random.seed(42)
    train_loader, val_loader, test_loader = make_loaders(x, y, train_idx, val_idx, test_idx)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    history = {"val_r": [], "val_mse": []}
    best_mse = float("inf")
    best_state = None
    patience_left = patience

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb.to(DEVICE))
            loss = nn.functional.mse_loss(pred, yb.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        vmse, vr, _ = evaluate(model, val_loader)
        history["val_r"].append(vr)
        history["val_mse"].append(vmse)
        sched.step(vmse)

        if vmse < best_mse - 1e-6:
            best_mse = vmse
            best_state = copy.deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    model.load_state_dict(best_state)
    tmse, tr, trho = evaluate(model, test_loader)
    history["test_mse"] = tmse
    history["test_r"] = tr
    history["test_rho"] = trho
    return model, history


print("Loading VAMOS + ViT embeddings...")
df = pd.read_csv(CSV_PATH)
clip_mat = np.load(f"{FEATURES_DIR}/clip_embeddings.npy").astype(np.float32)

required_cols = ["Img", "Memorability_Score"] + VA_COLS
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise KeyError(f"Missing columns in CSV: {missing_cols}")
if len(df) != len(clip_mat):
    raise ValueError(f"Row mismatch: csv={len(df)} embeddings={len(clip_mat)}")

valid_mask = df[["Memorability_Score"] + VA_COLS].notna().all(axis=1).values
if not np.all(valid_mask):
    n_drop = int((~valid_mask).sum())
    print(f"Dropping {n_drop} rows with missing Memorability/Valence/Arousal.")
    df = df.loc[valid_mask].reset_index(drop=True)
    clip_mat = clip_mat[valid_mask]

memscore = df["Memorability_Score"].values.astype(np.float32)
va_mat = df[VA_COLS].values.astype(np.float32)
emotion_scores = np.vstack([va_to_emotion_scores(v, a) for v, a in va_mat]).astype(np.float32)

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
emo_norm = norm_block(emotion_scores, train_idx)

results = {}


def run(tag, model, x, desc):
    print(f"\n{'='*60}\n{tag} | {desc}\ndim={x.shape[1]}")
    m, h = train_model(model, x, memscore, train_idx, val_idx, test_idx)
    print(f"test MSE={h['test_mse']:.5f}  r={h['test_r']:.4f}  rho={h['test_rho']:.4f}")
    results[tag] = {"model": m, "history": h, "desc": desc, "X": x}


run("A_vit_only", LinearProbe(VIT_DIM), clip_mat, "ViT only baseline")
X_b = np.concatenate([clip_mat, emotion_scores], axis=1)
run("B_vit_emotions_raw", LinearProbe(VIT_DIM + len(EMOTIONS)), X_b, "ViT + derived emotions (raw)")
X_c = np.concatenate([clip_norm, emo_norm], axis=1)
run("C_vit_emotions_norm", LinearProbe(VIT_DIM + len(EMOTIONS)), X_c, "ViT + derived emotions (normalized)")

print("\n" + "=" * 60)
print("LASSO ON DERIVED EMOTIONS")
X_lasso = X_c.astype(np.float64)
y_train = memscore[train_idx].astype(np.float64)
y_val = memscore[val_idx].astype(np.float64)
y_test = memscore[test_idx].astype(np.float64)

lasso = LassoCV(cv=5, max_iter=10000, n_alphas=120, random_state=42)
lasso.fit(X_lasso[train_idx], y_train)
lasso_pred = lasso.predict(X_lasso[test_idx])
lasso_r, _ = pearsonr(lasso_pred, y_test)
lasso_rho, _ = spearmanr(lasso_pred, y_test)
lasso_mse = float(np.mean((lasso_pred - y_test) ** 2))
print(f"Lasso: alpha={lasso.alpha_:.6f} MSE={lasso_mse:.5f} r={lasso_r:.4f} rho={lasso_rho:.4f}")

emo_coef = lasso.coef_[VIT_DIM:]
print("Derived emotion coefficients:")
for e, w in zip(EMOTIONS, emo_coef):
    print(f"  {e:<9} w={w:+.6f}")

if SHAP_AVAILABLE:
    print("\n" + "=" * 60)
    print("SHAP ANALYSIS (Lasso - derived emotions)")
    bg_size = min(300, len(train_idx))
    ts_size = min(500, len(test_idx))
    bg_local = np.random.choice(len(train_idx), bg_size, replace=False)
    ts_local = np.random.choice(len(test_idx), ts_size, replace=False)
    X_bg = X_lasso[train_idx][bg_local]
    X_eval = X_lasso[test_idx][ts_local]

    explainer = shap.LinearExplainer(lasso, X_bg)
    sv = explainer.shap_values(X_eval)
    if isinstance(sv, list):
        sv = sv[0]

    emo_mean_abs = np.abs(sv[:, VIT_DIM:]).mean(axis=0)
    vit_mean_abs = np.abs(sv[:, :VIT_DIM]).mean()
    print(f"Mean |SHAP| ViT: {vit_mean_abs:.6f}")
    for emo, val in zip(EMOTIONS, emo_mean_abs):
        print(f"  Mean |SHAP| {emo:<9}: {val:.6f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(EMOTIONS, emo_mean_abs, color=plt.cm.tab10(np.linspace(0, 1, len(EMOTIONS))), edgecolor="white")
    ax.axhline(vit_mean_abs, color="black", linestyle="--", linewidth=1.2, label=f"ViT mean={vit_mean_abs:.6f}")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title("Derived emotion contribution from VA ranges")
    ax.set_xticks(np.arange(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=25, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/shap_derived_emotions_weight.png", dpi=170)
    plt.show()

    plt.figure(figsize=(9, 4))
    shap.summary_plot(
        sv[:, VIT_DIM:],
        X_eval[:, VIT_DIM:],
        feature_names=EMOTIONS,
        show=False,
    )
    plt.title("SHAP beeswarm - derived emotion features")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/shap_derived_emotions_beeswarm.png", dpi=170, bbox_inches="tight")
    plt.show()
else:
    print("SHAP not installed; skipping SHAP plots. Install with: python3 -m pip install shap")

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
        "Experiment": "Lasso_derived_emotions",
        "Description": "LassoCV on ViT + derived emotion ranges",
        "Test MSE": round(lasso_mse, 5),
        "Test r": round(lasso_r, 4),
        "Test rho": round(lasso_rho, 4),
        "Dr": round(lasso_r - baseline_r, 4),
        "Drho": round(lasso_rho - baseline_rho, 4),
    }
)

table = pd.DataFrame(rows).sort_values("Test r", ascending=False)
print("\n" + "=" * 60)
print("FINAL RESULTS")
print(table.to_string(index=False))
table.to_csv(f"{SAVE_DIR}/results_va_to_emotions.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4))
for tag, color in [
    ("A_vit_only", "steelblue"),
    ("B_vit_emotions_raw", "orange"),
    ("C_vit_emotions_norm", "green"),
]:
    if tag in results:
        ax.plot(results[tag]["history"]["val_r"], label=tag, color=color, linewidth=1.6)
ax.set_xlabel("Epoch")
ax.set_ylabel("Val Pearson r")
ax.set_title("Validation curves - VA converted to emotion ranges")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{SAVE_DIR}/learning_curves_va_to_emotions.png", dpi=150)
plt.show()

print("\nAll outputs saved to:", SAVE_DIR)
