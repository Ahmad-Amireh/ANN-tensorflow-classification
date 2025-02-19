import tensorflow as tf 
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np 
import pickle
import streamlit as st

# Load Model & Encoders
model = load_model("model.h5")

with open("label_encoder_gender.pkl", "rb") as f:
    loader_encoder_gender = pickle.load(f)

with open("onehot_encoder_geo.pkl", "rb") as f:
    onehot_encoder_geo = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Streamlit App
st.title("🔮 Customer Churn Prediction")

# User Input
geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
gender = st.selectbox("👤 Gender", loader_encoder_gender.classes_)
age = st.slider("📅 Age", 18, 92)
balance = st.number_input("💰 Balance", min_value=0.0)
credit_score = st.number_input("💳 Credit Score", min_value=300, max_value=850)
estimated_salary = st.number_input("💵 Estimated Salary", min_value=0.0)
tenure = st.slider("📆 Tenure", 0, 10)
num_of_products = st.slider("📦 Number of Products", 1, 4)
has_credit_card = st.selectbox("💳 Has Credit Card?", [0, 1])
is_active_member = st.selectbox("🔄 Is Active Member?", [0, 1])

# Prepare Input Data
try:
    input_data = pd.DataFrame({
        "Gender": [loader_encoder_gender.transform([gender]).item()],  # Extract single value
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "Credit Score": [credit_score],
        "NumofProducts": [num_of_products],
        "HasCreditCard": [has_credit_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary]
    })

    # One-Hot Encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(onehot_encoder_geo.feature_names_in_)
    )

    # Combine Encoded Features with Input Data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale Input Data
    input_data_scaled = scaler.transform(input_data)

    # Predict Churn
    if st.button("🔍 Predict"):
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        # Display Result with Color Formatting
        st.write(f"📊 **Churn Probability:** `{prediction_proba:.2f}`")

        if prediction_proba > 0.5:
            st.markdown('<p style="color:red; font-size:20px;">🚨 The customer is **likely** to churn!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green; font-size:20px;">✅ The customer is **not likely** to churn.</p>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ An error occurred: {e}")
