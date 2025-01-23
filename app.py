import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Model Deployment", layout="wide")

# Application title
st.title("Card_Fraude_Detection")

# Function to load models
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

# List of available models
model_dir = "models"
if os.path.exists(model_dir):
    models = [f for f in os.listdir(model_dir) if f.endswith(".joblib")]
    if models:
        selected_model_name = st.selectbox("Select a model:", models)
        model_path = os.path.join(model_dir, selected_model_name)
        model = load_model(model_path)
        st.write(f"### Loaded model: {selected_model_name}")

        # Loading input data
        st.write("### Upload a CSV file for predictions:")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("### Preview of the uploaded data:", data.head())

            # Checking for available columns
            st.write("### Available columns:", data.columns.tolist())

            # Prediction interface
            if st.button("Predict"):
                try:
                    # Preprocessing (Standardization if necessary)
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data)

                    # Predictions
                    predictions = model.predict(data_scaled)
                    st.write("### Prediction results:")
                    st.write(pd.DataFrame({"Predictions": predictions}))
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    else:
        st.warning("No models found in the 'models' directory.")
else:
    st.error("The 'models' directory does not exist. Please create a directory named 'models' and place your models in .joblib format.")