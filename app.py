import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load trained objects
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Depression Prediction", layout="centered")

st.title("🧠 Depression Prediction System")
st.write("Logistic Regression based mental health screening")

st.markdown("### Enter details using sliders")

# --- SLIDERS ---
age = st.slider("Age", 10, 80, 25)
gender = st.slider("Gender (0 = Female, 1 = Male)", 0, 1, 0)
screen_time = st.slider("Daily Screen Time (hrs)", 0.0, 15.0, 5.0)
sleep_quality = st.slider("Sleep Quality (1–10)", 1, 10, 5)
stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
days_no_social = st.slider("Days Without Social Media", 0, 30, 3)
exercise = st.slider("Exercise Frequency (per week)", 0, 14, 3)
platform = st.slider("Social Media Platform (encoded)", 0, 5, 1)

# Collect inputs
input_data = np.array([
    age, gender, screen_time, sleep_quality,
    stress_level, days_no_social, exercise, platform
]).reshape(1, -1)

# --- PREDICTION ---
if st.button("Predict Depression Risk"):
    input_scaled = scaler.transform(input_data)

    
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔍 Prediction Result")

    # Custom threshold
threshold = 0.4   # you can explain this in viva

if prob >= threshold:
    st.error(f"High risk of depression (Probability: {prob:.2f})")
else:
    st.success(f"Low risk of depression (Probability: {prob:.2f})")


    # --- SIMPLE EXPLANATION ---
    st.markdown("### 🧠 Explanation of Prediction")

    if prob > 0.6:
        st.write(
            "The model predicts a higher risk mainly due to **high stress levels, "
            "poor sleep quality, and increased screen time**."
        )
    else:
        st.write(
            "The model predicts a lower risk due to **better sleep quality, "
            "regular exercise, and controlled screen usage**."
        )

    # --- FEATURE CONTRIBUTION PLOT (FAST & DIFFERENT BARS) ---
    st.markdown("### 📊 Feature Contributions")

    coefficients = model.coef_[0]
    contributions = coefficients * input_scaled[0]

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": contributions
    }).sort_values(by="Contribution", key=abs, ascending=False)

    fig, ax = plt.subplots()
    ax.barh(contrib_df["Feature"], contrib_df["Contribution"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Impact on Prediction")

    st.pyplot(fig)

# Disclaimer
st.markdown("---")
st.caption("⚠️ This tool is for educational purposes only and not a medical diagnosis.")



