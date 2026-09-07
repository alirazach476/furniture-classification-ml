"""Generate synthetic furniture-like image patches for classification demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DATA_DIR = Path(__file__).resolve().parent / "data" / "images"
CLASSES = ["chair", "table", "sofa", "bed", "cabinet"]

# Distinct color themes per class so a simple ML model can separate them.
PALETTES = {
    "chair": [(180, 90, 50), (210, 140, 90)],
    "table": [(90, 70, 40), (130, 100, 60)],
    "sofa": [(40, 70, 140), (80, 110, 180)],
    "bed": [(200, 200, 210), (230, 230, 235)],
    "cabinet": [(60, 60, 60), (110, 110, 110)],
}


def _draw_class(img: Image.Image, label: str, rng: np.random.Generator) -> None:
    draw = ImageDraw.Draw(img)
    c1, c2 = PALETTES[label]
    noise = tuple(int(x) for x in rng.integers(-20, 21, size=3))
    fill = tuple(max(0, min(255, a + b)) for a, b in zip(c1, noise))
    accent = tuple(max(0, min(255, a + b)) for a, b in zip(c2, noise))

    if label == "chair":
        draw.rectangle([20, 25, 44, 55], fill=fill)
        draw.rectangle([20, 55, 70, 70], fill=accent)
        draw.rectangle([20, 70, 28, 100], fill=fill)
        draw.rectangle([62, 70, 70, 100], fill=fill)
    elif label == "table":
        draw.rectangle([15, 45, 95, 60], fill=fill)
        draw.rectangle([22, 60, 30, 100], fill=accent)
        draw.rectangle([80, 60, 88, 100], fill=accent)
    elif label == "sofa":
        draw.rounded_rectangle([10, 40, 100, 85], radius=12, fill=fill)
        draw.rectangle([10, 40, 100, 58], fill=accent)
    elif label == "bed":
        draw.rectangle([10, 50, 100, 90], fill=fill)
        draw.rectangle([10, 35, 35, 55], fill=accent)
        draw.rectangle([10, 90, 18, 105], fill=(80, 80, 80))
        draw.rectangle([92, 90, 100, 105], fill=(80, 80, 80))
    else:  # cabinet
        draw.rectangle([25, 15, 85, 100], fill=fill)
        draw.line([55, 15, 55, 100], fill=accent, width=3)
        draw.ellipse([48, 52, 54, 58], fill=accent)
        draw.ellipse([56, 52, 62, 58], fill=accent)


def generate_dataset(samples_per_class: int = 120, size: int = 64, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    for label in CLASSES:
        out = DATA_DIR / label
        out.mkdir(parents=True, exist_ok=True)
        for i in range(samples_per_class):
            bg = tuple(int(x) for x in rng.integers(200, 255, size=3))
            img = Image.new("RGB", (size, size), bg)
            _draw_class(img, label, rng)
            # mild noise
            arr = np.array(img).astype(np.int16)
            arr += rng.integers(-8, 9, size=arr.shape)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(out / f"{label}_{i:04d}.png")
    print(f"Generated {samples_per_class * len(CLASSES)} images in {DATA_DIR}")


if __name__ == "__main__":
    generate_dataset()
