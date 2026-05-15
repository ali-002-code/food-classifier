"""
src/train.py
------------
Training loop with:
  - Per-epoch train loss / train accuracy
  - Per-epoch validation loss / validation accuracy / top-5 accuracy
  - Learning rate scheduling
  - Model checkpointing (best val accuracy + periodic)
  - Full history returned for plotting

Design notes
------------
The train/validate split within this file is deliberate: the functions are
small, single-purpose, and testable in isolation. The main `train()` function
orchestrates them without containing logic of its own — easier to debug.

Mixed precision (torch.cuda.amp) is enabled when a CUDA device is present.
It uses float16 for forward/backward passes and float32 for the optimiser step,
cutting memory usage roughly in half and increasing throughput on modern GPUs,
with no change to model accuracy.
"""

import os
import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple


# --------------------------------------------------------------------------- #
# Core step functions
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Run one full pass over the training set.

    Returns:
        average loss, top-1 accuracy (as a fraction, not percentage)
    """
    model.train()
    # model.train() activates Dropout and BatchNorm's running-stat updates.
    # model.eval() would freeze them — always switch modes explicitly.

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad()
        # zero_grad() before forward pass — PyTorch accumulates gradients
        # by default. Failing to zero them means each step adds to previous.

        with autocast(enabled=device.type == "cuda"):
            # autocast: run forward pass in float16 where safe
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        # scaler.scale() multiplies loss by a scale factor to prevent
        # float16 underflow in gradients, then unscales before the step.

        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """
    Evaluate model on the validation/test set.

    @torch.no_grad() disables gradient computation entirely — no backward
    graph is built, saving memory and time during inference.

    Returns:
        average loss, top-1 accuracy, top-5 accuracy
    """
    model.eval()

    total_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)

        # Top-1: is the highest-probability class correct?
        top1_preds = logits.argmax(dim=1)
        correct_top1 += (top1_preds == labels).sum().item()

        # Top-5: is the true label among the 5 highest-probability classes?
        # This is standard in ImageNet-style evaluations. For a dish app,
        # it tells you: "would the user's dish appear in the top 5 suggestions?"
        top5_preds = logits.topk(5, dim=1).indices   # (B, 5)
        correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

    avg_loss   = total_loss / total
    acc_top1   = correct_top1 / total
    acc_top5   = correct_top5 / total
    return avg_loss, acc_top1, acc_top5


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
) -> Dict[str, List]:
    """
    Full training run.

    Returns a history dict with per-epoch metrics for plotting:
        {
            "train_loss": [...],
            "train_acc":  [...],
            "val_loss":   [...],
            "val_acc":    [...],
            "val_acc5":   [...],
            "lr":         [...],
        }
    """
    device = torch.device(config["device"])
    model = model.to(device)

    # ------------------------------------------------------------------ #
    # Loss function
    # CrossEntropyLoss = log-softmax + negative log-likelihood.
    # It expects raw logits (not probabilities) and class indices (not
    # one-hot vectors). label_smoothing=0.1 slightly softens targets,
    # making the model less overconfident — regularisation effect.
    # ------------------------------------------------------------------ #
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ------------------------------------------------------------------ #
    # Optimiser — AdamW
    # Adam: adaptive per-parameter learning rates. Fast convergence.
    # Weight decay in Adam is technically wrong (it couples with the
    # adaptive scaling) — AdamW fixes this by decoupling L2 regularisation
    # from the gradient update. Always prefer AdamW over Adam + L2.
    # ------------------------------------------------------------------ #
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    # ------------------------------------------------------------------ #
    # Learning rate scheduler
    # CosineAnnealingLR: smoothly decays LR from lr to ~0 over all epochs.
    # This typically outperforms StepLR (sudden drops) by avoiding the
    # model "forgetting" structure it built before each step.
    # ------------------------------------------------------------------ #
    if config["lr_scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser, T_max=config["epochs"]
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser,
            step_size=config["lr_step_size"],
            gamma=config["lr_gamma"],
        )

    scaler = GradScaler(enabled=device.type == "cuda")

    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    history: Dict[str, List] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":  [], "val_acc5": [],
        "lr": [],
    }
    best_val_acc = 0.0

    print(f"\nTraining on: {device}")
    print(f"Model parameters: {model.count_parameters():,}\n")
    print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  "
          f"{'Val Loss':>8}  {'Val Acc':>7}  {'Val Acc5':>8}  "
          f"{'LR':>8}  {'Time':>6}")
    print("-" * 80)

    for epoch in range(1, config["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, scaler, device
        )
        val_loss, val_acc, val_acc5 = validate(
            model, test_loader, criterion, device
        )

        scheduler.step()
        current_lr = optimiser.param_groups[0]["lr"]

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_acc5"].append(val_acc5)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>8.2%}  "
            f"{val_loss:>8.4f}  {val_acc:>7.2%}  {val_acc5:>8.2%}  "
            f"{current_lr:>8.2e}  {elapsed:>5.1f}s"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            _save_checkpoint(model, optimiser, epoch, history, config,
                             filename="best_model.pt")
            print(f"         -> New best val accuracy: {best_val_acc:.2%}")

        # Periodic checkpoint
        if epoch % config["save_every"] == 0:
            _save_checkpoint(model, optimiser, epoch, history, config,
                             filename=f"checkpoint_epoch{epoch:03d}.pt")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.2%}")
    return history


def _save_checkpoint(model, optimiser, epoch, history, config, filename):
    path = os.path.join(config["checkpoint_dir"], filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimiser_state_dict": optimiser.state_dict(),
        "history": history,
        "config": config,
    }, path)
