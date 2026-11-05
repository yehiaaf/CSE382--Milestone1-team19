import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from src.config import path

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def save_fig(fig, name):
    target = path("outputs", "figures", name)
    fig.tight_layout()
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def plot_confusion(cm, name, title):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Not canceled", "Canceled"]).plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    return save_fig(fig, name)


def plot_roc(y, proba, name, title):
    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color=COLORS[0])
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    return save_fig(fig, name)


def plot_pred_vs_actual(y, pred, name, title):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y, pred, s=4, alpha=0.25, color=COLORS[0])
    top = max(y.max(), pred.max())
    ax.plot([0, top], [0, top], color=COLORS[3])
    ax.set_xlabel("Actual ADR")
    ax.set_ylabel("Predicted ADR (out-of-fold)")
    ax.set_title(title)
    return save_fig(fig, name)


def plot_residuals(y, pred, name, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.hist(y - pred, bins=80, color=COLORS[0])
    ax.axvline(0, color=COLORS[3])
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Bookings")
    ax.set_title(title)
    return save_fig(fig, name)


def plot_k_sweep(table, name):
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    axes[0].plot(table["k"], table["inertia"], marker="o")
    axes[0].set_title("Inertia (elbow)")
    axes[1].plot(table["k"], table["silhouette"], marker="o", color=COLORS[1])
    axes[1].set_title("Silhouette (higher is better)")
    axes[2].plot(table["k"], table["davies_bouldin"], marker="o", color=COLORS[2])
    axes[2].set_title("Davies-Bouldin (lower is better)")
    axes[3].plot(table["k"], table["calinski_harabasz"], marker="o", color=COLORS[4])
    axes[3].set_title("Calinski-Harabasz (higher is better)")
    for ax in axes:
        ax.set_xlabel("Number of clusters k")
    return save_fig(fig, name)


def plot_clusters(coords, labels, name, title, explained):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for c in np.unique(labels):
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=5, alpha=0.5, color=COLORS[c % len(COLORS)], label="Cluster %d" % c)
    ax.set_xlabel("PC1 (%.1f%% variance)" % (explained[0] * 100))
    ax.set_ylabel("PC2 (%.1f%% variance)" % (explained[1] * 100))
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=8)
    return save_fig(fig, name)