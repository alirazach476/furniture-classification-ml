"""Train a furniture image classifier using classical ML on image features."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from generate_data import CLASSES, DATA_DIR, generate_dataset

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "models"
REPORTS = BASE / "reports"


def load_xy(size: int = 32):
    if not DATA_DIR.exists() or not any(DATA_DIR.rglob("*.png")):
        generate_dataset()

    xs, ys = [], []
    for idx, label in enumerate(CLASSES):
        for path in sorted((DATA_DIR / label).glob("*.png")):
            img = Image.open(path).convert("RGB").resize((size, size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            # color histogram + flattened pixels (compact classical features)
            hist = []
            for c in range(3):
                h, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)
                hist.append(h)
            feat = np.concatenate([arr.flatten(), *hist])
            xs.append(feat)
            ys.append(idx)
    return np.asarray(xs), np.asarray(ys)


def train() -> dict:
    x, y = load_xy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=80, random_state=42)),
            ("clf", SVC(kernel="rbf", C=4.0, gamma="scale", probability=True)),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    report = classification_report(y_test, preds, target_names=CLASSES, digits=4)
    cm = confusion_matrix(y_test, preds)
    acc = float((preds == y_test).mean())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": CLASSES}, MODEL_DIR / "furniture_model.joblib")
    (REPORTS / "classification_report.txt").write_text(
        f"Accuracy: {acc:.4f}\n\n{report}", encoding="utf-8"
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title("Furniture Classification")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrix.png", dpi=140)
    plt.close()

    print(f"Accuracy: {acc:.4f}")
    print(report)
    return {"accuracy": acc, "report": report}


def predict_image(path: str | Path) -> dict:
    bundle = joblib.load(MODEL_DIR / "furniture_model.joblib")
    model = bundle["model"]
    classes = bundle["classes"]
    img = Image.open(path).convert("RGB").resize((32, 32))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    hist = []
    for c in range(3):
        h, _ = np.histogram(arr[:, :, c], bins=16, range=(0, 1), density=True)
        hist.append(h)
    feat = np.concatenate([arr.flatten(), *hist]).reshape(1, -1)
    proba = model.predict_proba(feat)[0]
    idx = int(np.argmax(proba))
    return {"label": classes[idx], "confidence": float(proba[idx]), "probabilities": dict(zip(classes, map(float, proba)))}


if __name__ == "__main__":
    train()
