#!/bin/bash

# Variables AWS
AWS_REGION="us-east-1"   # Remplace par ta région AWS
INSTANCE_ID="i-xxxxxxxxxxxxx"  # ID de ton instance EC2 (tu peux le récupérer avec l'AWS CLI)
IMAGE_NAME="fraud:latest"
CONTAINER_NAME="fraud_app"

# Connexion SSH et déploiement sur EC2
ssh -o StrictHostKeyChecking=no ec2-user@<Public_IP_de_ton_instance> << EOF
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
    docker pull $IMAGE_NAME
    docker run -d -p 8501:8501 --name $CONTAINER_NAME $IMAGE_NAME
EOF
