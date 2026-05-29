# DishNet: Food Dish Image Classifier

A custom convolutional neural network trained from scratch to classify images of food dishes into 101 categories, using the [Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset.

Built as a solo project to explore CNN architecture design, training dynamics, and the engineering decisions that go into a production-ready image classifier - without relying on pretrained models or transfer learning.

---

## Results

| Metric | Value |
|---|---|
| Top-1 Accuracy (test set) | ~55% |
| Top-5 Accuracy (test set) | ~82% |
| Training epochs | 30 |
| Model parameters | ~7.3M |
| Training time (GPU) | ~3 hours |

> **Context:** A fine-tuned ResNet50 (pretrained on ImageNet) achieves ~85% top-1 on Food-101. A CNN trained from scratch in 30 epochs is expected to land in the 50–60% range. The gap reflects what transfer learning buys you- not a flaw in the architecture.

---

## Architecture: DishNet

```
Input: (B, 3, 128, 128)

ConvBlock x5:
  Conv2d(3x3, bias=False) -> BatchNorm2d -> ReLU
  Conv2d(3x3, bias=False) -> BatchNorm2d -> ReLU
  MaxPool2d(2x2)

Channel progression: 3 -> 32 -> 64 -> 128 -> 256 -> 512
Spatial progression: 128 -> 64 -> 32 -> 16 -> 8 -> 4

Global Average Pooling: (B, 512, 4, 4) -> (B, 512)

Classifier head:
  Linear(512, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 101)

Output: (B, 101) logits
```

### Key design decisions

**Two 3x3 convolutions per block instead of one 5x5**

Two stacked 3x3 convolutions have an effective receptive field of 5x5, with fewer parameters (`2 * 9 * C^2` vs `25 * C^2`) and an extra non-linearity between them. This is the core insight from VGGNet - deeper is better than wider, and smaller kernels are more parameter-efficient.

**BatchNorm after every Conv layer**

BatchNorm normalises the distribution of activations between layers. This allows higher learning rates without diverging, acts as mild regularisation, and reduces sensitivity to weight initialisation. Applied after Conv and before the activation (pre-activation order debated, this follows the original BN paper).

**`bias=False` in Conv layers**

Redundant with BatchNorm. BatchNorm has its own learnable bias parameter (β), so the Conv bias is completely overwritten. Omitting it saves parameters and avoids a trivially redundant computation.

**Global Average Pooling instead of Flatten**

After the 5th Conv block, spatial size is 4x4. Using `Flatten` then `Linear` would give a head of size `512*16 = 8192`, making the first FC layer `8192*256 ≈ 2M parameters` - expensive and prone to overfitting. GAP collapses each feature map to a single number ("how much of this feature is present anywhere"), reducing the head input to 512. The head becomes `512*256 ≈ 131k parameters` - a 15x reduction.

**AdamW over Adam**

Standard Adam's weight decay implementation is technically incorrect - it couples L2 regularisation with the adaptive gradient scaling, causing the effective regularisation to vary per parameter. AdamW decouples them, applying weight decay directly to weights before the gradient update. For the same weight decay value, AdamW generalises better.

**Cosine Annealing LR schedule**

Cosine annealing smoothly decays the learning rate from `lr_max` to ~0 over all epochs. Compared to step decay (sudden drops), it avoids the model "forgetting" structure built before each step, and the gradual decay lets the optimiser settle into a sharper minimum near the end of training.

**Label smoothing (ε = 0.1)**

Instead of hard targets (1 for correct class, 0 for all others), label smoothing uses `1 - ε` for the correct class and `ε / (C-1)` for the rest. This prevents the model becoming overconfident on training examples, improving calibration and generalisation. Particularly useful for Food-101 which has visually similar classes.

**Kaiming initialisation for Conv layers**

Kaiming (He) initialisation sets initial weights to `N(0, sqrt(2/fan_in))`. The factor of 2 is specifically derived to account for ReLU zeroing half its inputs - without it, variance would shrink through each layer and signals would vanish in deep networks. Xavier initialisation (used for Linear layers here) assumes a linear activation and uses `sqrt(2/(fan_in + fan_out))`.

---

## Project Structure

```
food-classifier/
├── main.py               # Entry point: trains and evaluates
├── config.py             # All hyperparameters in one place
├── requirements.txt
├── src/
│   ├── model.py          # DishNet architecture
│   ├── dataset.py        # Food-101 data loading and augmentation
│   ├── train.py          # Training loop with mixed precision
│   ├── evaluate.py       # Confusion matrix, per-class F1, learning curves
│   └── inference.py      # Single-image inference + FastAPI server
└── results/
    ├── confusion_matrix.png
    ├── learning_curves.png
    └── per_class_metrics.csv
```

---

## Setup and Training

```bash
git clone https://github.com/YOUR_USERNAME/food-classifier
cd food-classifier
pip install -r requirements.txt

# Full training run (downloads Food-101 ~5GB on first run)
python main.py

# Quick debug run (10% of data, 3 epochs)
python main.py --subset 0.1 --epochs 3
```

**Hardware notes:**
- GPU recommended (NVIDIA). Training time ~3 hours on a T4 (Google Colab).
- CPU: feasible with `--subset 0.2` for a proof-of-concept run.
- Mixed precision (float16) is automatically enabled on CUDA devices.

---

## Data Augmentation

Training augmentations are chosen to simulate realistic variation in dish photography:

| Transform | Reason |
|---|---|
| RandomCrop (from 147px to 128px) | Simulates different framing / zoom |
| RandomHorizontalFlip | Dish looks the same mirrored |
| ColorJitter (brightness, contrast, saturation, hue) | Simulates lighting variation |
| RandomRotation (±15°) | Simulates slight camera tilt |

Validation uses only CenterCrop + Normalise - augmentation is never applied to validation data, as it would introduce noise into the metric.

---

## Evaluation

```bash
# Evaluate a checkpoint and generate all plots
python src/evaluate.py --checkpoint results/checkpoints/best_model.pt
```

Outputs:
- `confusion_matrix.png` - the 20 most confused class pairs (full 101x101 is illegible)
- `per_class_metrics.csv` - precision, recall, F1 per class, sorted by F1
- `learning_curves.png` - train/val loss, top-1/top-5 accuracy, LR schedule over epochs

---

## Dishboxd Integration

This model powers optional food detection in [Dishboxd](https://github.com/YOUR_USERNAME/dishboxd), replacing the Claude Vision API call with a local inference endpoint.

**Deploy as a FastAPI server:**

```bash
pip install fastapi uvicorn python-multipart
# Uncomment the `app` variable at the bottom of src/inference.py, then:
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

**Call from your React Native app:**

```typescript
const classifyDish = async (imageUri: string) => {
  const formData = new FormData();
  formData.append("file", {
    uri: imageUri,
    type: "image/jpeg",
    name: "dish.jpg",
  } as any);

  const response = await fetch("http://YOUR_SERVER:8000/classify", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();
  // result.top_label: "Pizza"
  // result.top_confidence: 0.87
  // result.is_food: true
  // result.predictions: [{ label: "Pizza", confidence: 0.87 }, ...]
  return result;
};
```

---

## What I learned / would do differently

**Worked well:**
- GAP made a meaningful difference - the model regularises better than a Flatten + large FC layer
- Label smoothing noticeably reduced training/validation accuracy gap
- AdamW with cosine annealing converged more smoothly than Adam + step decay in early experiments

**Limitations of a from-scratch CNN:**
- Feature reuse is limited - pretrained models have seen 1.2M diverse images; Food-101 training only gives 750 per class
- Deeper architectures (ResNet, EfficientNet) benefit disproportionately from pretraining because residual connections allow more effective gradient flow

**If I continued this project:**
- Self-supervised pretraining (SimCLR or DINO) on unlabelled food images before fine-tuning on Food-101 - would recover most of the gap to pretrained ImageNet models without using labelled ImageNet data
- Knowledge distillation: train a smaller student model from this network for faster mobile inference in Dishboxd

---

## References

- [Food-101 Dataset](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) - Bossard et al., 2014
- [Very Deep Convolutional Networks (VGGNet)](https://arxiv.org/abs/1409.1556) - Simonyan & Zisserman, 2015 - motivation for stacking 3x3 convs
- [Batch Normalisation](https://arxiv.org/abs/1502.03167) - Ioffe & Szegedy, 2015
- [Delving Deep into Rectifiers (Kaiming init)](https://arxiv.org/abs/1502.01852) - He et al., 2015
- [Decoupled Weight Decay Regularisation (AdamW)](https://arxiv.org/abs/1711.05101) - Loshchilov & Hutter, 2019
- [When Does Label Smoothing Help?](https://arxiv.org/abs/1906.02629) - Müller et al., 2019
