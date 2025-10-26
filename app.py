import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Load Trained Model

model_path = "model.pkl"

try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("⚠ model.pkl not found! Place it in the same folder as this app.")
    st.stop()


# App Header

st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")
st.title("🏡 House Price Prediction App")
st.markdown("""
Welcome to the *Interactive House Price Predictor*!  
Provide property details and get an *instant price prediction* with model comparison insights.
""")


# Sidebar Inputs

st.sidebar.header("Enter Property Details")

# Example features — match these to your dataset columns
def user_input_features():
    OverallQual = st.sidebar.slider("Overall Quality (1–10)", 1, 10, 5)
    GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", 500, 5000, 1500)
    GarageCars = st.sidebar.slider("Garage Capacity (Cars)", 0, 5, 2)
    TotalBsmtSF = st.sidebar.number_input("Total Basement Area (sq ft)", 0, 3000, 800)
    YearBuilt = st.sidebar.slider("Year Built", 1900, 2025, 2000)
    FullBath = st.sidebar.slider("Full Bathrooms", 0, 5, 2)
    LotArea = st.sidebar.number_input("Lot Area (sq ft)", 1000, 20000, 8000)

    data = {
        "OverallQual": OverallQual,
        "GrLivArea": GrLivArea,
        "GarageCars": GarageCars,
        "TotalBsmtSF": TotalBsmtSF,
        "YearBuilt": YearBuilt,
        "FullBath": FullBath,
        "LotArea": LotArea
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader("📋 Entered Property Details")
st.write(input_df)


# Align Columns to Model Input

# Some models (like pipelines) have a method 'feature_names_in_'
try:
    model_features = model.feature_names_in_
    input_df = input_df.reindex(columns=model_features, fill_value=0)
except AttributeError:
    st.info("ℹ Model does not have feature_names_in_. Proceeding with input features only.")


# Predict Button

if st.button("Predict House Price"):
    try:
        prediction = model.predict(input_df)
        st.success(f"🏠 *Predicted House Price:* ${prediction[0]:,.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed due to: {e}")


# Feature Importance Visualization

st.subheader("📊 Feature Influence")

try:
    if hasattr(model, "feature_importances_"):
        features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else input_df.columns
        importance = model.feature_importances_

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(features, importance)
        ax.set_xlabel("Feature Importance")
        ax.set_title("Feature Importance Visualization")
        st.pyplot(fig)
    else:
        st.info("ℹ This model does not support feature importance visualization.")
except Exception as e:
    st.warning(f"⚠ Could not display feature importance: {e}")


# Footer

st.markdown("---")
st.markdown("Developed by *Sandeep Kumar Jena* | 5th Semester ML Internship Project (2025)")