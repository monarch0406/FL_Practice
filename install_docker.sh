#!/bin/bash

echo "🚀 Updating system..."
sudo apt update -y

echo "🐳 Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh

echo "👤 Adding user to docker group..."
sudo usermod -aG docker $USER

echo "🧩 Installing Docker Compose plugin..."
sudo apt install docker-compose-plugin -y

echo "✅ Verifying installation..."
docker --version
docker compose version

echo "🎉 Done! Please logout and log back in or run: newgrp docker"
