import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('best_model.pkl')

st.title("🏠 House Price Prediction App")
st.write("Enter details below to predict house price:")

# Example features (replace with your dataset columns)
# Make sure these match your training data!
col1, col2 = st.columns(2)
with col1:
    feature1 = st.number_input("Feature 1 (e.g., LotArea)", min_value=0)
    feature2 = st.number_input("Feature 2 (e.g., OverallQual)", min_value=0)
with col2:
    feature3 = st.number_input("Feature 3 (e.g., YearBuilt)", min_value=0)
    feature4 = st.number_input("Feature 4 (e.g., GrLivArea)", min_value=0)

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'LotArea': [feature1],
    'OverallQual': [feature2],
    'YearBuilt': [feature3],
    'GrLivArea': [feature4]
})

if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Predicted House Price: ${prediction:,.2f}")
