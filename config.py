"""
config.py
---------
Central configuration for DishNet training.

Keeping all hyperparameters here (not scattered across files) means you can
reproduce any experiment by logging this single dict, and changing behaviour
never requires hunting through source files.
"""

import torch

CONFIG = {
    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    "data_dir": "./data",           # torchvision will download Food-101 here
    "image_size": 128,              # resize all images to 128x128
                                    # 224 is more standard but 128 trains ~3x
                                    # faster and still gives meaningful results
    "num_classes": 101,
    "num_workers": 4,               # parallel data loading threads

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    "batch_size": 64,
    "epochs": 30,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,           # L2 regularisation applied inside AdamW
    "lr_scheduler": "cosine",       # "cosine" | "step"
    "lr_step_size": 10,             # used only if scheduler == "step"
    "lr_gamma": 0.1,                # used only if scheduler == "step"

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    "dropout_rate": 0.5,

    # ------------------------------------------------------------------ #
    # Misc
    # ------------------------------------------------------------------ #
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "./results/checkpoints",
    "results_dir": "./results",
    "save_every": 5,                # save a checkpoint every N epochs
}
