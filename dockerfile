# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le contenu local dans le conteneur
COPY . .

# Copier le dossier models
COPY Models/ /app/Models/




# Copier le dossier apps à partir de scripts/apps
COPY scripts/ /app/scripts/
#COPY scripts/* /app/scripts/
#COPY scripts/.* /app/scripts/


# COPY app.py ./

# Exposer le port sur lequel l'application s'exécute
EXPOSE 8502

# Commande par défaut pour exécuter l'application Streamlit
CMD ["streamlit", "run", "/app/scripts/apps/app.py", "--server.port=8502", "--server.address=0.0.0.0"]


