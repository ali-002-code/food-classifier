"""
src/inference.py
----------------
Drop-in inference module for Dishboxd integration.

This replaces the Claude Vision API call for food detection with a local
PyTorch model forward pass. Benefits:
  - No API cost per inference
  - Lower latency (no network round trip)
  - Works offline

Usage in Dishboxd (React Native / Expo)
----------------------------------------
You would deploy this as a FastAPI endpoint (see below) and call it from
the React Native app the same way you currently call the Claude API.

Or for a pure Python backend, import classify_dish_image directly.

Quick start
-----------
    from src.inference import DishClassifier

    classifier = DishClassifier(
        checkpoint_path="results/checkpoints/best_model.pt",
        device="cpu"   # or "cuda" if available
    )

    result = classifier.classify(image_path="path/to/dish.jpg", top_k=5)
    # result = {"predictions": [{"label": "pizza", "confidence": 0.87}, ...], "is_food": True}
"""

import json
import os
import torch
from torchvision import transforms
from PIL import Image
from typing import Union

from src.model import DishNet
from src.dataset import IMAGENET_MEAN, IMAGENET_STD


# These are the 101 Food-101 class names in index order.
# Stored here so the inference module is self-contained (no dataset needed at runtime).
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari",
    "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad",
    "grilled_cheese_sandwich", "grilled_salmon", "guacamole", "gyoza", "hamburger",
    "hot_and_sour_soup", "hot_dog", "huevos_rancheros", "hummus", "ice_cream",
    "lasagna", "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese",
    "macarons", "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
    "oysters", "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib", "pulled_pork_sandwich",
    "ramen", "ravioli", "red_velvet_cake", "risotto", "samosa", "sashimi",
    "scallops", "seaweed_salad", "shrimp_and_grits", "spaghetti_bolognese",
    "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare", "waffles",
]


class DishClassifier:
    """
    Stateful inference wrapper. Load once, call many times.
    Loading the model on every request would be very slow.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        class_names: list = None,
    ):
        self.device = torch.device(device)
        self.class_names = class_names or FOOD101_CLASSES

        # Load model
        self.model = DishNet(num_classes=len(self.class_names))

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Inference transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize(148),         # 128 * 1.15
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        print(f"DishClassifier loaded on {self.device}")

    @torch.no_grad()
    def classify(
        self,
        image: Union[str, "PIL.Image.Image"],
        top_k: int = 5,
        confidence_threshold: float = 0.05,
    ) -> dict:
        """
        Classify a dish image.

        Args:
            image: file path string or PIL Image
            top_k: return top k predictions
            confidence_threshold: minimum confidence to include a prediction.
                                  If the top-1 confidence is below this, flag
                                  as potentially not food.

        Returns:
            {
                "predictions": [{"label": "pizza", "confidence": 0.87}, ...],
                "is_food": bool,   # True if model is reasonably confident
                "top_label": "pizza",
                "top_confidence": 0.87,
            }
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze(0)

        top_probs, top_indices = probs.topk(min(top_k, len(self.class_names)))

        predictions = [
            {
                "label": self.class_names[idx.item()].replace("_", " ").title(),
                "confidence": round(prob.item(), 4),
            }
            for idx, prob in zip(top_indices, top_probs)
        ]

        top_confidence = predictions[0]["confidence"]
        is_food = top_confidence >= confidence_threshold

        return {
            "predictions": predictions,
            "is_food": is_food,
            "top_label": predictions[0]["label"],
            "top_confidence": top_confidence,
        }


# --------------------------------------------------------------------------- #
# FastAPI server — deploy this to replace the Claude Vision API call
# --------------------------------------------------------------------------- #

def create_fastapi_app(checkpoint_path: str, device: str = "cpu"):
    """
    Creates a FastAPI app for serving DishNet predictions.

    Deploy with: uvicorn src.inference:app --host 0.0.0.0 --port 8000

    Then in Dishboxd, replace:
        const result = await anthropic.messages.create({ ... vision call ... })
    with:
        const result = await fetch("http://your-server:8000/classify", {
            method: "POST",
            body: formData,   // FormData with image file
        })
    """
    try:
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import JSONResponse
        import io
    except ImportError:
        raise ImportError("Install fastapi and uvicorn: pip install fastapi uvicorn python-multipart")

    app = FastAPI(title="DishNet API", version="1.0")
    classifier = DishClassifier(checkpoint_path=checkpoint_path, device=device)

    @app.post("/classify")
    async def classify_dish(file: UploadFile = File(...), top_k: int = 5):
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = classifier.classify(image, top_k=top_k)
        return JSONResponse(content=result)

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "DishNet", "classes": len(FOOD101_CLASSES)}

    return app


# To run the FastAPI server:
# Uncomment and configure the lines below, then run:
#     uvicorn src.inference:app --reload
#
# app = create_fastapi_app(
#     checkpoint_path="results/checkpoints/best_model.pt",
#     device="cpu",
# )
