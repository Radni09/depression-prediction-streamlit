import streamlit as st
import numpy as np
import joblib

# Load saved objects
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Depression Prediction", layout="centered")

st.title("🧠 Depression Prediction System")
st.write("Logistic Regression based mental health screening")

# Input section
inputs = []
for feature in feature_names:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

# Prediction
if st.button("Predict"):
    X = np.array(inputs).reshape(1, -1)
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    st.subheader("Result")
    st.write("Prediction:", "Depressed" if pred == 1 else "Not Depressed")
    st.write("Probability of Depression:", round(prob, 3))



