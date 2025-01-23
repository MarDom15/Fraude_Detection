#!/bin/bash

# Variables AWS
PUBLIC_IP="16.170.228.250"
IMAGE_NAME="fraud:latest"
CONTAINER_NAME="fraud_app"

# Connexion SSH et déploiement
ssh -o StrictHostKeyChecking=no ec2-user@$PUBLIC_IP << EOF
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
    docker pull $IMAGE_NAME
    docker run -d -p 8501:8501 --name $CONTAINER_NAME $IMAGE_NAME
EOF
