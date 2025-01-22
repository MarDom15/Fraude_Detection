import os
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image

# Chemin du modèle dans le dossier 'Models'
MODEL_PATH = 'Models/final_model.h5'

# Vérification du modèle
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Le modèle n'a pas été trouvé dans le chemin : {MODEL_PATH}")

# Chargement du modèle
model = load_model(MODEL_PATH)

# Classes des maladies
CLASS_LABELS = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "squamous cell carcinoma",
    "vascular lesion"
]

def preprocess_image(img_path):
    """Préparer l'image pour la prédiction."""
    img = load_img(img_path, target_size=(224, 224))  # Mise à l'échelle de l'image
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le batch
    img_array = img_array / 255.0  # Normalisation des pixels (0 à 1)
    return img_array

# Interface Streamlit
st.title("Skin Disease Classification")
st.write("Téléversez une image de peau pour prédire la maladie.")

# Chargement de l'image par l'utilisateur
uploaded_file = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Afficher l'image téléversée
    st.image(uploaded_file, caption="Image Téléversée", use_column_width=True)

    # Sauvegarder l'image localement
    img_path = os.path.join("static/uploads", uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Préparer l'image et effectuer une prédiction
    st.write("Traitement de l'image...")
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)[0]

    # Résultats des prédictions
    results = {CLASS_LABELS[i]: round(pred * 100, 2) for i, pred in enumerate(predictions)}
    
    # Afficher les résultats
    st.subheader("Résultats des prédictions :")
    for label, confidence in results.items():
        st.write(f"**{label}:** {confidence}%")
