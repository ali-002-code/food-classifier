"""
src/model.py
------------
DishNet: a custom CNN for food dish classification.

Architecture overview
---------------------
Input: (B, 3, 128, 128)

5 convolutional blocks, each following the pattern:
    Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d

Doubling channels at each block (32 -> 64 -> 128 -> 256 -> 512) is standard
practice: as spatial resolution falls, channel depth rises to preserve
representational capacity. Think of it as trading spatial precision for
semantic richness.

After block 5: Global Average Pooling (GAP)
    - Converts (B, 512, H, W) to (B, 512) regardless of spatial size
    - Massively fewer parameters than a Flatten + Linear approach
    - Each of the 512 feature maps gets reduced to a single average,
      so each output neuron represents "how much of this feature is
      present anywhere in the image"

Classifier head:
    Linear(512, 256) -> ReLU -> Dropout -> Linear(256, 101)

Interview talking points
------------------------
Q: Why BatchNorm?
A: Normalises the distribution of activations between layers, which:
   (a) lets you use higher learning rates without diverging,
   (b) acts as a mild regulariser,
   (c) reduces sensitivity to weight initialisation.
   Applied AFTER Conv and BEFORE the activation.

Q: Why two Conv layers per block before pooling?
A: Each 3x3 conv on its own has a receptive field of 3x3. Two stacked 3x3
   convs have an effective receptive field of 5x5 but with fewer parameters
   than a single 5x5 conv (2*9*C^2 vs 25*C^2) and an extra non-linearity
   between them (more expressive). This is the insight from VGGNet.

Q: Why Global Average Pooling over Flatten?
A: Flatten followed by a fully-connected layer would be (512 * H * W) inputs.
   At H=W=4 that is 512*16=8192 inputs, making the head 8192*256 = 2M params.
   GAP collapses spatial dims to 1x1 first: head becomes 512*256 = 131k params.
   Also more robust to spatial shift in the input image.

Q: Why Dropout only in the head, not the conv blocks?
A: Spatial dropout in conv layers (dropping entire channels) can help but
   regular dropout is less effective on feature maps because neighbouring
   pixels are strongly correlated. The dense head is where standard dropout
   has most impact.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two convolutional layers with BatchNorm and ReLU, followed by MaxPool.

    Pattern: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # First conv layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # bias=False because BatchNorm has its own learnable bias (beta).
            # Including both is redundant and wastes parameters.
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Second conv layer — same spatial size, same channels
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # Halve spatial dimensions
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DishNet(nn.Module):
    """
    Custom CNN for food dish classification.

    Args:
        num_classes: number of output categories (101 for Food-101)
        dropout_rate: probability of zeroing a head neuron during training
    """

    def __init__(self, num_classes: int = 101, dropout_rate: float = 0.5):
        super().__init__()

        # -------------------------------------------------------------- #
        # Feature extractor: 5 conv blocks
        # Input:  (B, 3,   128, 128)
        # After block 1: (B, 32,  64,  64)
        # After block 2: (B, 64,  32,  32)
        # After block 3: (B, 128, 16,  16)
        # After block 4: (B, 256,  8,   8)
        # After block 5: (B, 512,  4,   4)
        # -------------------------------------------------------------- #
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
        )

        # -------------------------------------------------------------- #
        # Global Average Pooling: (B, 512, 4, 4) -> (B, 512, 1, 1)
        # -------------------------------------------------------------- #
        self.gap = nn.AdaptiveAvgPool2d(1)
        # AdaptiveAvgPool2d(1) is preferred over a fixed-size pool because
        # it works for any input resolution — useful if you later want to
        # run inference on non-128 images.

        # -------------------------------------------------------------- #
        # Classifier head
        # -------------------------------------------------------------- #
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
            # No softmax here: CrossEntropyLoss expects raw logits and
            # applies log-softmax internally (numerically more stable).
        )

        # -------------------------------------------------------------- #
        # Weight initialisation
        # -------------------------------------------------------------- #
        self._initialise_weights()

    def _initialise_weights(self):
        """
        Kaiming (He) initialisation for Conv layers, Xavier for Linear.

        Why Kaiming for Conv? It accounts for the ReLU activation that
        follows — specifically designed to keep variance stable through
        ReLU non-linearities by scaling by sqrt(2/fan_in).

        Why Xavier for Linear? Linear layers feed into ReLU too but
        Kaiming and Xavier perform similarly here; Xavier is conventional.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)   # gamma = 1 (scale)
                nn.init.zeros_(module.bias)    # beta  = 0 (shift)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)       # conv blocks
        x = self.gap(x)            # (B, 512, 1, 1)
        x = x.flatten(start_dim=1) # (B, 512)
        x = self.classifier(x)     # (B, num_classes)
        return x

    def count_parameters(self) -> int:
        """Convenience method — useful to quote in interviews/README."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check — run this file directly to verify shapes
    model = DishNet(num_classes=101)
    dummy = torch.randn(4, 3, 128, 128)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")
