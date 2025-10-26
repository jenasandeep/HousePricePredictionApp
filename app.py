import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt


# Load the trained model artifac

model_path = "model.pkl"

try:
    with open(model_path, "rb") as file:
        artifact = pickle.load(file)
        pipeline = artifact['pipeline']
        feature_columns = artifact['feature_columns']
        numeric_features = artifact['numeric_features']
        categorical_features = artifact['categorical_features']
        model_name = artifact['model_name']
except FileNotFoundError:
    st.error("⚠ Model file not found! Please ensure 'model.pkl' is in the same folder.")
    st.stop()

# App Configuration

st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")
st.title("🏡 House Price Prediction App")
st.markdown(f"""
Welcome to the *House Price Prediction* tool!  
**Model loaded:** {model_name}  
Provide property details below and get an *instant price prediction* 💰.
""")


# User Input Section

st.sidebar.header("Enter Property Details")

# Automatically create input fields based on feature types
user_input = {}
for feature in feature_columns:
    if feature in numeric_features:
        user_input[feature] = st.sidebar.number_input(
            feature, min_value=0, value=0, step=1
        )
    elif feature in categorical_features:
        # For simplicity, free text input; you can replace with selectbox if categories are known
        user_input[feature] = st.sidebar.text_input(feature, value="")

input_df = pd.DataFrame([user_input])


# Display user input

st.subheader("📋 Entered Property Details")
st.write(input_df)


# Make Prediction

if st.button("Predict House Price"):
    try:
        prediction = pipeline.predict(input_df)[0]
        st.success(f"🏠 Predicted House Price: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")


# Visualization Section
st.subheader("📊 Feature Influence (Example Visualization)")
st.markdown("Feature importance chart shows how each input affects the predicted price.")

try:
    # Extract final estimator from pipeline
    estimator = pipeline
    if hasattr(pipeline, "steps"):
        estimator = pipeline.steps[-1][1]

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_

        # Get the transformed feature names
        transformed_feature_names = []

        if hasattr(pipeline, "named_steps") and "preprocessor" in pipeline.named_steps:
            preprocessor = pipeline.named_steps["preprocessor"]

            # Numeric features
            if hasattr(preprocessor, "transformers"):
                for name, transformer, cols in preprocessor.transformers:
                    if transformer == "drop":
                        continue
                    if hasattr(transformer, "get_feature_names_out"):
                        transformed_feature_names.extend(transformer.get_feature_names_out(cols))
                    else:
                        transformed_feature_names.extend(cols)
        else:
            # Fallback: use original feature_columns
            transformed_feature_names = feature_columns

        # Make sure length matches importance
        if len(transformed_feature_names) != len(importance):
            st.warning("⚠ Feature names and importances length mismatch. Showing top features only.")
            min_len = min(len(transformed_feature_names), len(importance))
            transformed_feature_names = transformed_feature_names[:min_len]
            importance = importance[:min_len]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(transformed_feature_names, importance)
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model.")
except Exception as e:
    st.warning(f"Could not display feature importance: {e}")
