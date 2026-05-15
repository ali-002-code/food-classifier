"""
main.py
-------
Entry point for training DishNet.

Usage
-----
# Full training run (downloads Food-101 on first run ~5GB):
    python main.py

# Quick debug run using only 10% of data:
    python main.py --subset 0.1 --epochs 3

# Custom config overrides:
    python main.py --batch_size 32 --lr 5e-4 --epochs 50
"""

import argparse
import os
import random
import numpy as np
import torch

from config import CONFIG
from src.model import DishNet
from src.dataset import get_dataloaders
from src.train import train
from src.evaluate import (
    collect_predictions,
    plot_confusion_matrix,
    save_classification_report,
    plot_learning_curves,
)


def set_seed(seed: int):
    """
    Fix all sources of randomness for reproducibility.

    PyTorch, NumPy, and Python's random module each have independent RNGs.
    Setting all three ensures the same results across runs on the same hardware.
    Note: full determinism on GPU also requires CUBLAS_WORKSPACE_CONFIG (see below).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional: strict determinism (slower)
    # torch.use_deterministic_algorithms(True)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main(args):
    set_seed(CONFIG["seed"])

    # Allow CLI to override config values
    if args.epochs:
        CONFIG["epochs"] = args.epochs
    if args.batch_size:
        CONFIG["batch_size"] = args.batch_size
    if args.lr:
        CONFIG["learning_rate"] = args.lr

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    train_loader, test_loader, class_names = get_dataloaders(
        data_dir=CONFIG["data_dir"],
        image_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        subset_fraction=args.subset,
    )

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = DishNet(
        num_classes=CONFIG["num_classes"],
        dropout_rate=CONFIG["dropout_rate"],
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    history = train(model, train_loader, test_loader, CONFIG)

    # ------------------------------------------------------------------ #
    # Evaluate (load best checkpoint)
    # ------------------------------------------------------------------ #
    print("\nLoading best model checkpoint for final evaluation...")
    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG["device"])
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device(CONFIG["device"])
    model = model.to(device)

    print("Collecting predictions on test set...")
    preds, labels = collect_predictions(model, test_loader, device)

    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    plot_confusion_matrix(
        preds, labels, class_names,
        save_path=os.path.join(CONFIG["results_dir"], "confusion_matrix.png"),
    )

    save_classification_report(
        preds, labels, class_names,
        save_path=os.path.join(CONFIG["results_dir"], "per_class_metrics.csv"),
    )

    plot_learning_curves(
        history,
        save_dir=CONFIG["results_dir"],
    )

    print("\nAll results saved to ./results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DishNet food classifier")
    parser.add_argument("--subset", type=float, default=1.0,
                        help="Fraction of dataset to use (default: 1.0)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override config epochs")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override config batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override config learning rate")
    args = parser.parse_args()
    main(args)
