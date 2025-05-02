import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris

# --- Load trained model and scaler using pickle ---
try:
    with open("knn_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError as e:
    st.error(f"❌ Error: {e}. Please ensure 'knn_model.pkl' and 'scaler.pkl' are in the app directory.")
    st.stop()

# --- Load Iris dataset to get feature ranges and class names ---
iris = load_iris()
X = iris.data
target_names = iris.target_names

# --- Streamlit App UI ---
st.set_page_config(page_title="Iris Classifier", layout="centered")
st.title("🌸 Iris Flower Classification with KNN")
st.markdown("""
Welcome! This interactive app uses a trained **K-Nearest Neighbors (KNN)** model to predict the species of an Iris flower 
based on your input of sepal and petal dimensions. Adjust the sliders and hit **Predict**!
""")

# --- User Input via Sliders ---
sepal_length = st.slider("Sepal Length (cm)", float(X[:, 0].min()), float(X[:, 0].max()), step=0.1)
sepal_width  = st.slider("Sepal Width (cm)",  float(X[:, 1].min()), float(X[:, 1].max()), step=0.1)
petal_length = st.slider("Petal Length (cm)", float(X[:, 2].min()), float(X[:, 2].max()), step=0.1)
petal_width  = st.slider("Petal Width (cm)",  float(X[:, 3].min()), float(X[:, 3].max()), step=0.1)

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# --- Predict Button ---
if st.button("🔍 Predict"):
    try:
        # Input summary
        st.subheader("📌 Your Input")
        st.write(f"- Sepal Length: **{sepal_length} cm**")
        st.write(f"- Sepal Width: **{sepal_width} cm**")
        st.write(f"- Petal Length: **{petal_length} cm**")
        st.write(f"- Petal Width: **{petal_width} cm**")

        # Preprocess and predict
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)

        # Output
        st.success(f"✅ Predicted Class: **{target_names[prediction].capitalize()}**")
        proba_df = pd.DataFrame(prediction_proba, columns=[name.capitalize() for name in target_names])
        st.subheader("📊 Prediction Probabilities")
        st.dataframe(proba_df.style.highlight_max(axis=1, color='lightgreen'))

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
