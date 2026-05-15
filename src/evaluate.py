"""
src/evaluate.py
---------------
Post-training evaluation tools:
  - Full confusion matrix (saved as PNG)
  - Per-class precision, recall, F1 (saved as CSV)
  - Learning curve plots (loss and accuracy vs epoch)
  - Single-image inference utility

Run this file directly after training:
    python src/evaluate.py --checkpoint results/checkpoints/best_model.pt
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works in headless environments)
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #

@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Run inference over the full loader and return (all_preds, all_labels)."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    save_path: str,
    top_n: int = 20,
):
    """
    Plot a confusion matrix for the top_n most confused classes.

    101 classes makes a full 101x101 matrix illegible. Instead we:
      1. Find the top_n classes with the most misclassifications.
      2. Plot the submatrix for only those classes.
    This is far more useful for diagnosing what the model is struggling with.

    Args:
        top_n: number of most-confused classes to show
    """
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

    # Compute per-class misclassification count (off-diagonal sum per row)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = cm / row_sums.clip(min=1)   # normalised by true class count

    # Find classes with lowest recall (worst-classified)
    per_class_recall = np.diag(cm_norm)
    worst_indices = np.argsort(per_class_recall)[:top_n]
    worst_indices = sorted(worst_indices)

    sub_cm    = cm_norm[np.ix_(worst_indices, worst_indices)]
    sub_names = [class_names[i] for i in worst_indices]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(sub_cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Recall (fraction)")

    ax.set(
        xticks=np.arange(top_n),
        yticks=np.arange(top_n),
        xticklabels=sub_names,
        yticklabels=sub_names,
        xlabel="Predicted class",
        ylabel="True class",
        title=f"Confusion Matrix — {top_n} Most Confused Classes (normalised by row)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    # Annotate cells with values
    for i in range(top_n):
        for j in range(top_n):
            val = sub_cm[i, j]
            colour = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=colour)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


# --------------------------------------------------------------------------- #
# Per-class metrics
# --------------------------------------------------------------------------- #

def save_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list,
    save_path: str,
):
    """
    Save per-class precision, recall, F1, and support to a CSV.

    Precision = TP / (TP + FP)  — of all images predicted as class X, what fraction were X?
    Recall    = TP / (TP + FN)  — of all images actually class X, what fraction did we catch?
    F1        = harmonic mean of precision and recall

    For a food classifier: recall matters more (you want to detect the dish),
    while precision matters for showing suggestions (you don't want wrong suggestions).
    """
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    df = pd.DataFrame(report).T
    df = df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    df = df.sort_values("f1-score", ascending=False)
    df.to_csv(save_path)
    print(f"Per-class metrics saved to: {save_path}")

    # Print summary to console
    overall_acc = (preds == labels).mean()
    print(f"\nOverall accuracy:        {overall_acc:.4f} ({overall_acc:.2%})")
    print(f"Macro-avg F1:            {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted-avg F1:         {report['weighted avg']['f1-score']:.4f}")

    # Best and worst 5 classes
    print("\nTop 5 best classified classes:")
    print(df.head(5)[["precision", "recall", "f1-score", "support"]].to_string())
    print("\nTop 5 worst classified classes:")
    print(df.tail(5)[["precision", "recall", "f1-score", "support"]].to_string())


# --------------------------------------------------------------------------- #
# Learning curves
# --------------------------------------------------------------------------- #

def plot_learning_curves(history: dict, save_dir: str):
    """Plot train/val loss and accuracy over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Loss ---
    axes[0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   linewidth=2)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    train_acc_pct = [a * 100 for a in history["train_acc"]]
    val_acc_pct   = [a * 100 for a in history["val_acc"]]
    val_acc5_pct  = [a * 100 for a in history["val_acc5"]]

    axes[1].plot(epochs, train_acc_pct, label="Train Top-1", linewidth=2)
    axes[1].plot(epochs, val_acc_pct,   label="Val Top-1",   linewidth=2)
    axes[1].plot(epochs, val_acc5_pct,  label="Val Top-5",   linewidth=2, linestyle="--")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # --- Learning Rate ---
    axes[2].plot(epochs, history["lr"], linewidth=2, color="green")
    axes[2].set(title="Learning Rate Schedule", xlabel="Epoch", ylabel="LR")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("DishNet Training History", fontsize=13, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "learning_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Learning curves saved to: {save_path}")


# --------------------------------------------------------------------------- #
# Single-image inference (used by Dishboxd integration)
# --------------------------------------------------------------------------- #

def predict_single_image(
    model: nn.Module,
    image_tensor: torch.Tensor,
    class_names: list,
    device: torch.device,
    top_k: int = 5,
) -> list:
    """
    Run inference on a single pre-processed image tensor.

    Args:
        image_tensor: (3, H, W) normalised tensor
        top_k: number of top predictions to return

    Returns:
        List of (class_name, probability) tuples, sorted by probability desc.
    """
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))   # (1, num_classes)
        probs  = torch.softmax(logits, dim=1).squeeze(0)       # (num_classes,)

    top_probs, top_indices = probs.topk(top_k)
    return [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main():
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from src.model import DishNet
    from src.dataset import get_dataloaders
    from config import CONFIG

    parser = argparse.ArgumentParser(description="Evaluate DishNet checkpoint")
    parser.add_argument("--checkpoint", type=str, default="results/checkpoints/best_model.pt")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of test set to evaluate on (1.0 = full)")
    args = parser.parse_args()

    device = torch.device(CONFIG["device"])

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DishNet(num_classes=CONFIG["num_classes"],
                    dropout_rate=CONFIG["dropout_rate"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Data
    _, test_loader, class_names = get_dataloaders(
        CONFIG["data_dir"], CONFIG["image_size"],
        CONFIG["batch_size"], CONFIG["num_workers"],
        subset_fraction=args.subset,
    )

    # Predictions
    print("\nRunning inference over test set...")
    preds, labels = collect_predictions(model, test_loader, device)

    # Outputs
    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    plot_confusion_matrix(
        preds, labels, class_names,
        save_path=os.path.join(CONFIG["results_dir"], "confusion_matrix.png"),
    )

    save_classification_report(
        preds, labels, class_names,
        save_path=os.path.join(CONFIG["results_dir"], "per_class_metrics.csv"),
    )

    # Plot learning curves from checkpoint history
    if "history" in checkpoint:
        plot_learning_curves(
            checkpoint["history"],
            save_dir=CONFIG["results_dir"],
        )


if __name__ == "__main__":
    main()
