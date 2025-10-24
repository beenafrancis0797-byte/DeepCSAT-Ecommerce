import streamlit as st
st.set_page_config(page_title="DeepCSAT Predictor", page_icon="🤖", layout="centered")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
import numpy as np, os, json, joblib

# -----------------------------
# 📁 Paths
# -----------------------------
MODEL_DIR = r"C:/Users/beena/deepcsat_ann_artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "deepcsat_ann_model.keras")


# -----------------------------
# ⚙️ Load Model and Artifacts
# -----------------------------
@st.cache_resource
def load_model_and_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    meta = json.load(open(os.path.join(MODEL_DIR, "meta.json")))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    
    # Recreate TextVectorization
    vcfg = os.path.join(MODEL_DIR, "vectorizer_config.json")
    vectorizer = None
    if os.path.exists(vcfg):
        vec_conf = json.load(open(vcfg))
        vectorizer = TextVectorization.from_config(vec_conf)
        dummy_vocab = ["<pad>", "<start>", "order", "arrived", "late", "item", "damaged", "good", "bad"]
        vectorizer.set_vocabulary(dummy_vocab)
    
    return model, meta, scaler, vectorizer

model, meta, scaler, vectorizer = load_model_and_artifacts()
text_col = meta.get("text_col")
numeric_cols = meta.get("numeric_cols", [])

# -----------------------------
# 🧠 Prediction Function
# -----------------------------
def predict_csat(text, numeric_dict):
    num_vals = np.array([[numeric_dict.get(c, 0.0) for c in numeric_cols]],
                        dtype="float32") if numeric_cols else np.zeros((1, 0))
    if scaler is not None and num_vals.shape[1] > 0:
        num_vals = scaler.transform(num_vals)

    text_input = tf.constant([[text]])
    pred = model.predict({"text_input": text_input, "num_input": num_vals}, verbose=0)[0][0]
    return float(pred)

# -----------------------------
# 🎨 Streamlit UI
# -----------------------------

st.title("🤖 DeepCSAT – Customer Satisfaction Predictor")
st.markdown("**Predict customer satisfaction scores using AI!**")

feedback = st.text_area("✍️ Enter customer feedback text:", height=120)

# Input numeric fields dynamically
numeric_dict = {}
st.markdown("### 📊 Enter numeric features (optional):")
cols = st.columns(3)
for i, col in enumerate(numeric_cols):
    with cols[i % 3]:
        numeric_dict[col] = st.number_input(f"{col}", value=0.0)

if st.button("🔍 Predict CSAT"):
    if not feedback.strip():
        st.warning("Please enter some text before predicting.")
    else:
        pred = predict_csat(feedback, numeric_dict)
        st.success(f"Predicted CSAT Score: **{pred:.2f} / 5.0**")
        
        if pred >= 4:
            st.markdown("😃 **High Satisfaction** – Great customer experience!")
        elif pred >= 3:
            st.markdown("🙂 **Moderate Satisfaction** – Could be improved.")
        else:
            st.markdown("😟 **Low Satisfaction** – Needs immediate attention.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit + TensorFlow")
