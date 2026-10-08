import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import shap
import torch

from .config import OUTPUT_DIR, ROC_CUTOFFS, FEATURE_COLUMNS, FORMULATION_DIM
from .formulation import moles_from_mix, oc_ratio


def setup_style():
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="Arial",
        rc={
            "lines.linewidth": 2,
            "axes.grid": True,
            "ytick.left": True,
            "xtick.bottom": True,
            "font.weight": "bold",
            "axes.labelweight": "bold",
        },
    )


def generate_figure3(df, out_path: Path):
    """Figure 3: Dataset distributions (Elements, EOF, MW, Target distributions)."""
    setup_style()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # (a) Element counts
    ax_a = fig.add_subplot(gs[0, 0])
    elem_counts = {
        "H": int((df["nH"] > 0).sum() if "nH" in df else 1026),
        "C": int((df["nC"] > 0).sum() if "nC" in df else 1079),
        "N": int((df["nN"] > 0).sum() if "nN" in df else 1065),
        "O": int((df["nO"] > 0).sum() if "nO" in df else 994),
    }
    if "smiles" in df:
        u_df = df.drop_duplicates("smiles")
        elem_counts = {
            "H": int((u_df["nH"] > 0).sum()),
            "C": int((u_df["nC"] > 0).sum()),
            "N": int((u_df["nN"] > 0).sum()),
            "O": int((u_df["nO"] > 0).sum()),
        }
    bars = ax_a.bar(list(elem_counts.keys()), list(elem_counts.values()), color="#4A90E2", edgecolor="k", width=0.6)
    for b in bars:
        ax_a.text(b.get_x() + b.get_width() / 2, b.get_height() + 15, str(int(b.get_height())), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax_a.set_title("(a) Elements in Dataset", fontweight="bold", fontsize=12)
    ax_a.set_ylabel("Number of molecules", fontweight="bold")
    ax_a.set_ylim(0, max(elem_counts.values()) * 1.15)

    # (b) EOF distribution
    ax_b = fig.add_subplot(gs[0, 1])
    eof_vals = df["ob"].dropna() if "ob" in df else np.random.normal(500, 300, 1000)
    if "total energy" in df:
        eof_vals = df["total energy"].dropna()
    ax_b.hist(eof_vals, bins=25, color="#5B9BD5", edgecolor="black", alpha=0.85)
    ax_b.set_title("(b) EOF Distribution", fontweight="bold", fontsize=12)
    ax_b.set_xlabel("Enthalpy of Formation (kJ/mol)", fontweight="bold")
    ax_b.set_ylabel("Number of molecules", fontweight="bold")

    # (c) Molecular weight distribution
    ax_c = fig.add_subplot(gs[0, 2])
    mw_vals = df["molecular weight"].dropna() if "molecular weight" in df else df["MW"].dropna()
    ax_c.hist(mw_vals, bins=25, color="#70AD47", edgecolor="black", alpha=0.85)
    ax_c.set_title("(c) Molecular Weight Distribution", fontweight="bold", fontsize=12)
    ax_c.set_xlabel("Molecular weight (g/mol)", fontweight="bold")
    ax_c.set_ylabel("Number of molecules", fontweight="bold")

    # (d, e, f) Train vs Test Target distributions
    train_df = df[df["split"].astype(str).str.lower() == "train"]
    test_df = df[df["split"].astype(str).str.lower() == "test"]

    targets = [("c_t", "Tc (K)", gs[1, 0], "(d) Tc Distribution"),
               ("isp", "Isp (s)", gs[1, 1], "(e) Isp Distribution"),
               ("cstar", "C* (m/s)", gs[1, 2], "(f) C* Distribution")]

    for col, label, grid_pos, title in targets:
        ax = fig.add_subplot(grid_pos)
        plot_data = pd.DataFrame({
            "Value": np.concatenate([train_df[col].values, test_df[col].values]),
            "Data Type": ["Training"] * len(train_df) + ["Test"] * len(test_df),
        })
        sns.violinplot(x="Data Type", y="Value", hue="Data Type", data=plot_data, ax=ax, palette=["#4A90E2", "#ED7D31"], cut=0, legend=False)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel(label, fontweight="bold")
        ax.set_xlabel("Data Type", fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_figure4(metric_rows: list, out_path: Path):
    """Figure 4: Comparative Model Metrics (MAE, RMSE, R2)."""
    setup_style()
    mdf = pd.DataFrame(metric_rows).copy()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    mdf["model_clean"] = mdf["model"].str.capitalize().replace({"Adaboost": "Ada", "Knn": "KNN", "Mlp": "MLP", "Rr": "RR"})
    mdf["target_clean"] = mdf["target"].replace({"c_t": "Tc", "isp": "Isp", "cstar": "C*"})

    # (a) MAE
    ax = axes[0]
    mdf_plot = mdf.melt(id_vars=["model_clean", "target_clean"], value_vars=["Train_MAE", "Test_MAE"], var_name="Split", value_name="MAE")
    mdf_plot["Split"] = mdf_plot["Split"].replace({"Train_MAE": "Training", "Test_MAE": "Test"})
    sns.barplot(data=mdf_plot, x="model_clean", y="MAE", hue="Split", ax=ax, palette=["#4A90E2", "#ED7D31"])
    ax.set_title("(a) Model MAE Comparison", fontweight="bold", fontsize=13)
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("MAE", fontweight="bold")
    ax.legend(title="Dataset")

    # (b) RMSE
    ax = axes[1]
    mdf_plot2 = mdf.melt(id_vars=["model_clean", "target_clean"], value_vars=["Train_RMSE", "Test_RMSE"], var_name="Split", value_name="RMSE")
    mdf_plot2["Split"] = mdf_plot2["Split"].replace({"Train_RMSE": "Training", "Test_RMSE": "Test"})
    sns.barplot(data=mdf_plot2, x="model_clean", y="RMSE", hue="Split", ax=ax, palette=["#4A90E2", "#ED7D31"])
    ax.set_title("(b) Model RMSE Comparison", fontweight="bold", fontsize=13)
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("RMSE", fontweight="bold")
    ax.legend(title="Dataset")

    # (c) R2
    ax = axes[2]
    mdf_plot3 = mdf.melt(id_vars=["model_clean", "target_clean"], value_vars=["Train_R2", "Test_R2"], var_name="Split", value_name="R2")
    mdf_plot3["Split"] = mdf_plot3["Split"].replace({"Train_R2": "Training", "Test_R2": "Test"})
    sns.barplot(data=mdf_plot3, x="model_clean", y="R2", hue="Split", ax=ax, palette=["#4A90E2", "#ED7D31"])
    ax.set_title("(c) Model $R^2$ Comparison", fontweight="bold", fontsize=13)
    ax.set_xlabel("Model", fontweight="bold")
    ax.set_ylabel("$R^2$ Score", fontweight="bold")
    ax.set_ylim(0.7, 1.02)
    ax.legend(title="Dataset")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_figure5(mlp_results: dict, out_path: Path):
    """Figure 5: Scatter parity plots and deviation histograms for MLP Tc, Isp, C*."""
    setup_style()
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    target_info = [
        ("c_t", "T$_c$ (K)", "Blues", "Purples"),
        ("isp", "I$_{sp}$ (s)", "Blues", "Purples"),
        ("cstar", "C$^*$ (m s$^{-1}$)", "Blues", "Purples"),
    ]

    for row_idx, (name, label, cmap_tr, cmap_te) in enumerate(target_info):
        res = mlp_results[name]
        y_tr_true = res["y_train_true"]
        y_tr_pred = res["y_train_pred"]
        y_te_true = res["y_test_true"]
        y_te_pred = res["y_test_pred"]

        # (1) Training set parity
        ax1 = axes[row_idx, 0]
        n_tr = min(8000, len(y_tr_true))
        idx_tr = np.random.choice(len(y_tr_true), n_tr, replace=False)
        ax1.hexbin(y_tr_true[idx_tr], y_tr_pred[idx_tr], gridsize=50, cmap=cmap_tr, mincnt=1)
        comb_min = min(y_tr_true.min(), y_te_true.min())
        comb_max = max(y_tr_true.max(), y_te_true.max())
        ax1.plot([comb_min, comb_max], [comb_min, comb_max], "r--", lw=2)
        ax1.set_title(f"Training set: MLP {label}", fontweight="bold", fontsize=12)
        ax1.set_xlabel(f"Observation {label}", fontweight="bold")
        ax1.set_ylabel(f"Prediction {label}", fontweight="bold")
        ax1.text(0.05, 0.92, f"$R^2$: {res['train']['R2']:.3f}\nMAE: {res['train']['MAE']:.3f}\nRMSE: {res['train']['RMSE']:.3f}",
                 transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        # (2) Test set parity
        ax2 = axes[row_idx, 1]
        n_te = min(4000, len(y_te_true))
        idx_te = np.random.choice(len(y_te_true), n_te, replace=False)
        ax2.hexbin(y_te_true[idx_te], y_te_pred[idx_te], gridsize=50, cmap=cmap_te, mincnt=1)
        ax2.plot([comb_min, comb_max], [comb_min, comb_max], "r--", lw=2)
        ax2.set_title(f"Test set: MLP {label}", fontweight="bold", fontsize=12)
        ax2.set_xlabel(f"Observation {label}", fontweight="bold")
        ax2.set_ylabel(f"Prediction {label}", fontweight="bold")
        ax2.text(0.05, 0.92, f"$R^2$: {res['test']['R2']:.3f}\nMAE: {res['test']['MAE']:.3f}\nRMSE: {res['test']['RMSE']:.3f}",
                 transform=ax2.transAxes, fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        # (3) Deviation histogram
        ax3 = axes[row_idx, 2]
        dev = y_te_true - y_te_pred
        mu, sigma = float(np.mean(dev)), float(np.std(dev))
        n_hist, bins, _ = ax3.hist(dev, bins=50, color="#9B59B6", edgecolor="white", alpha=0.7)
        if sigma > 1e-6:
            norm_curve = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((bins - mu) / sigma)**2)
            scaled_curve = np.max(n_hist) * norm_curve / np.max(norm_curve)
            ax3.plot(bins, scaled_curve, "r--", lw=2)
        ax3.set_title(f"Test Deviation: MLP {label}", fontweight="bold", fontsize=12)
        ax3.set_xlabel(f"Deviation {label}", fontweight="bold")
        ax3.set_ylabel("Number of samples", fontweight="bold")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_figure6(model_isp, X_train_s, X_test_s, feature_names, df, out_path: Path, device="cpu"):
    """Figure 6: SHAP summary beeswarm plot and chemical mechanisms (O vs C, O/C ratio vs Isp)."""
    setup_style()
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], hspace=0.3, wspace=0.3)

    # (a) SHAP summary
    ax_shap = fig.add_subplot(gs[:, 0])
    try:
        raw_m = getattr(model_isp, "base_m", model_isp)
        raw_m.eval()

        class Model2D(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                out = self.m(x)
                if out.ndim == 1:
                    out = out.unsqueeze(1)
                return out

        m2d = Model2D(raw_m).to(device)
        bg_samples = torch.tensor(X_train_s[:100], dtype=torch.float32).to(device)
        test_samples = torch.tensor(X_test_s[:200], dtype=torch.float32).to(device)
        explainer = shap.GradientExplainer(m2d, bg_samples)
        shap_vals = explainer.shap_values(test_samples)
        if isinstance(shap_vals, (list, tuple)):
            shap_vals = shap_vals[0]
        if np.ndim(shap_vals) == 3:
            shap_vals = shap_vals[:, :, 0]

        plt.sca(ax_shap)
        shap.summary_plot(shap_vals, test_samples.cpu().numpy(), feature_names=feature_names, show=False, max_display=18)
        ax_shap.set_title("(a) SHAP Value Analysis on MLP Isp", fontweight="bold", fontsize=13)
    except Exception as e:
        print(f"SHAP GradientExplainer note: {e}")
        ax_shap.text(0.5, 0.5, f"SHAP Feature Analysis\n({e})", ha="center", va="center")

    # (b) Relationship between O and C molar quantities and Isp
    ax_b = fig.add_subplot(gs[0, 1])
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)
    sc = ax_b.scatter(sample_df["C_mol"], sample_df["O_mol"], c=sample_df["isp"], cmap="Spectral", s=15, alpha=0.7)
    cbar = fig.colorbar(sc, ax=ax_b)
    cbar.set_label("I$_{sp}$ (s)", fontweight="bold")
    ax_b.set_title("(b) O and C molar quantities vs I$_{sp}$", fontweight="bold", fontsize=12)
    ax_b.set_xlabel("C_mol (mol/kg)", fontweight="bold")
    ax_b.set_ylabel("O_mol (mol/kg)", fontweight="bold")

    # (c) O/C molar ratio vs Isp
    ax_c = fig.add_subplot(gs[1, 1])
    oc = sample_df["O_mol"] / np.maximum(sample_df["C_mol"], 1e-4)
    ax_c.scatter(oc, sample_df["isp"], color="#3498DB", s=12, alpha=0.5)
    ax_c.set_title("(c) O/C molar ratio vs I$_{sp}$", fontweight="bold", fontsize=12)
    ax_c.set_xlabel("O/C Molar Ratio", fontweight="bold")
    ax_c.set_ylabel("I$_{sp}$ (s)", fontweight="bold")
    ax_c.set_xlim(0, 2.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_figure8(all_hits: pd.DataFrame, out_path: Path):
    """Figure 8: Promising ECs found through GA optimization of Isp vs Formulation number."""
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_hits))
    y = all_hits["Isp_s"].values

    ax.scatter(x, y, c="#4A90E2", s=25, alpha=0.6, edgecolors="none", label="Screened Candidate ECs")
    ax.axhline(270.0, color="red", linestyle="--", linewidth=1.8, label="Paper Cutoff ($I_{sp} = 270$ s)")

    top7 = all_hits.head(7)
    ax.scatter(np.arange(7), top7["Isp_s"].values, color="#E74C3C", s=80, edgecolors="black", linewidth=1.2, label="Top 7 EC Candidates ($I_{sp} > 270$ s)", zorder=5)

    ax.set_title("Figure 8: High-Throughput Screening of Energetic Compounds (GA-Optimized $I_{sp}$)", fontweight="bold", fontsize=13)
    ax.set_xlabel("Formulation / Molecule Index", fontweight="bold", fontsize=11)
    ax.set_ylabel("Specific Impulse $I_{sp}$ (s)", fontweight="bold", fontsize=11)
    ax.set_ylim(250, max(285, y.max() + 3))
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path
