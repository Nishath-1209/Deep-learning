import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    layout="wide"
)

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(
    "model/titanic_ann_model.h5"
)

# =========================
# HEADER SECTION
# =========================

st.title("🚢 Titanic Survival Prediction System")

st.subheader(
    "Deep Learning Based Passenger Survival Prediction"
)

st.markdown("---")

# =========================
# PROJECT DESCRIPTION
# =========================

st.markdown("""
## 📘 Project Description

This application predicts whether a passenger would survive during the Titanic disaster using:

- Artificial Neural Networks (ANN)
- TensorFlow/Keras
- Streamlit Deployment

The model uses:
- Passenger Class
- Age
- Fare

to predict survival probability.
""")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================

st.header("🧾 Passenger Information")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=50.0
    )

# =========================
# PREPROCESSING
# =========================

# Manual normalization ranges
# Pclass: 1-3
# Age: 1-80
# Fare: 0-512

pclass_norm = (pclass - 1) / (3 - 1)
age_norm = (age - 1) / (80 - 1)
fare_norm = fare / 512

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

# =========================
# PREDICTION
# =========================

if st.button("Predict Survival"):

    prediction = model.predict(input_data, verbose=0)

    probability = prediction[0][0]

    st.markdown("---")

    st.header("📊 Prediction Result")

    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prediction",
            result
        )

    with col2:
        st.metric(
            "Survival Probability",
            f"{probability*100:.2f}%"
        )

    # =========================
    # VISUALIZATION
    # =========================

    st.subheader("📈 Probability Visualization")

    labels = ['Survived', 'Not Survived']

    values = [
        probability,
        1 - probability
    ]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    st.pyplot(fig)
