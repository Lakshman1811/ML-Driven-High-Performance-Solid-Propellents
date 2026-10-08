from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

from .config import OUTPUT_DIR, ROC_CUTOFFS

TARGET_LABEL = {"isp": "Isp (s)", "c_t": "Tc (K)", "cstar": "C* (m/s)"}


def plot_loss_curves(histories: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, name in zip(axes, ["isp", "c_t", "cstar"]):
        h = histories[name]
        ax.plot(h["epoch"], h["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        ax.plot(h["epoch"], h["test_loss"], label="Test Loss", color="#ff7f0e", lw=2)
        ax.set_title(f"MLP {TARGET_LABEL[name]} Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("MSE Loss", fontsize=11)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_roc_curves(results: dict, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rows = []
    for ax, name in zip(axes, ["isp", "c_t", "cstar"]):
        y_true = np.asarray(results[name]["y_test_true"])
        y_score = np.asarray(results[name]["y_test_pred"])
        cutoff = ROC_CUTOFFS.get(name, 270.0)
        labels = (y_true >= cutoff).astype(int)
        if labels.min() == labels.max():
            # Fallback to 85th percentile if cutoff is out of range
            cutoff = float(np.percentile(y_true, 85))
            labels = (y_true >= cutoff).astype(int)
            print(f"Adjusted cutoff for {name} to 85th percentile ({cutoff:.1f}) to generate valid ROC curve")

        fpr, tpr, _ = roc_curve(labels, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Random (AUC = 0.500)")
        ax.set_xlabel("False Positive Rate (FPR)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (TPR)", fontsize=11, fontweight="bold")
        ax.set_title(f"ROC {TARGET_LABEL[name]}  (Cutoff > {cutoff:g})", fontsize=12, fontweight="bold")
        ax.text(
            0.52, 0.22,
            f"AUC = {roc_auc:.4f}\n(Cutoff > {cutoff:g})\nPositives: {int(labels.sum())}/{len(labels)}",
            fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4f8", edgecolor="#1f77b4", alpha=0.9),
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        rows.append({"target": name, "cutoff": cutoff, "AUC": float(roc_auc), "n_positive": int(labels.sum())})
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path, rows

