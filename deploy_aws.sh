#!/bin/bash

# Variables AWS
PUBLIC_IP="16.170.228.250"
IMAGE_NAME="fraud:latest"  # Le nom de l'image Docker
CONTAINER_NAME="fraud_app"
PRIVATE_KEY_PATH="C:/Users/marti/Desktop/dataof/Fraude_Detection/martial_domche_Mum.pem"  # Remplacez par le chemin correct de votre clé

# Connexion SSH et déploiement
ssh -i $PRIVATE_KEY_PATH -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP << EOF
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
    docker pull $IMAGE_NAME
    docker run -d -p 8502:8502 --name $CONTAINER_NAME $IMAGE_NAME
EOF
