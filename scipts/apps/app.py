import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import pickle
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Model Deployment", layout="wide")

# Application title
st.title("Card Fraud Detection")

# Function to load models
@st.cache_resource
def load_model(model_path):
    # Charger un modèle au format .pkl
    with open(model_path, 'rb') as file:
        return pickle.load(file)

# Define the directory for models
model_dir = "/app/models"  # Path where models are stored inside the container

# Debugging paths: Check if the model directory exists
st.write("### Checking if model directory exists...")
if os.path.exists(model_dir):
    st.write(f"Model directory found: {model_dir}")
    st.write("Models available:", os.listdir(model_dir))
else:
    st.error(f"Model directory not found: {model_dir}")


# Check if the models directory exists
if os.path.exists(model_dir):
    # List all `.pkl` files in the models directory
    models = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    
    if models:
        # Dropdown menu to select a model
        selected_model_name = st.selectbox("Select a model:", models)
        
        # Construct the full path to the selected model
        model_path = os.path.join(model_dir, selected_model_name)
        
        # Load the selected model
        model = load_model(model_path)
        st.write(f"### Loaded model: {selected_model_name}")
        
        # Uploading input data
        st.write("### Upload a CSV file for predictions:")
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file:
            # Load the uploaded data
            data = pd.read_csv(uploaded_file)
            st.write("### Preview of the uploaded data:", data.head())

            # Display the available columns in the data
            st.write("### Available columns:", data.columns.tolist())

            # Prediction interface
            if st.button("Predict"):
                try:
                    # Preprocessing (Standardization if necessary)
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data)

                    # Make predictions
                    predictions = model.predict(data_scaled)
                    st.write("### Prediction results:")
                    st.write(pd.DataFrame({"Predictions": predictions}))
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
    else:
        st.warning("No models found in the 'models' directory.")
else:
    st.error("The 'models' directory does not exist. Please create a directory named 'models' and place your models in `.pkl` format.")
