"""Streamlit UI for furniture classification."""

from pathlib import Path

import streamlit as st
from PIL import Image

from generate_data import CLASSES, DATA_DIR
from train import MODEL_DIR, predict_image, train

st.set_page_config(page_title="Furniture Classification", page_icon="🪑", layout="centered")
st.title("Furniture Classification")
st.caption("SVM + PCA on color/shape features — chair, table, sofa, bed, cabinet")

if not (MODEL_DIR / "furniture_model.joblib").exists():
    if st.button("Generate data & train"):
        with st.spinner("Training..."):
            metrics = train()
        st.success(f"Accuracy: {metrics['accuracy']:.2%}")
else:
    st.success("Model ready.")

uploaded = st.file_uploader("Upload furniture image", type=["png", "jpg", "jpeg"])
if uploaded and (MODEL_DIR / "furniture_model.joblib").exists():
    tmp = Path("tmp_upload.png")
    Image.open(uploaded).convert("RGB").save(tmp)
    st.image(str(tmp), width=220)
    result = predict_image(tmp)
    st.metric("Prediction", result["label"])
    st.write(f"Confidence: {result['confidence']:.2%}")
    st.bar_chart(result["probabilities"])

st.subheader("Sample gallery")
cols = st.columns(len(CLASSES))
for col, label in zip(cols, CLASSES):
    sample = next((DATA_DIR / label).glob("*.png"), None) if (DATA_DIR / label).exists() else None
    with col:
        st.caption(label)
        if sample:
            st.image(str(sample), use_container_width=True)
