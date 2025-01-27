#!/bin/bash

# Minikube 상태 확인
STATUS=$(minikube status --format='{{.Host}}')
if [ "$STATUS" == "Running" ]; then
  echo "Minikube is running."
else
  echo "Error: Minikube is not running."
  exit 1
fi

read -p "Enter the project root directory: " PROJECT_ROOT

# PROJECT_ROOT 유효성 검증
if [ -d "$PROJECT_ROOT" ]; then
    echo "Project root directory is set to: $PROJECT_ROOT"
else
    echo "Error: Directory '$PROJECT_ROOT' does not exist."
    exit 1
fi

eval $(minikube docker-env)

cd "$PROJECT_ROOT/train_mnist"
docker build -t mnist-train:v0.21 .

cd "$PROJECT_ROOT/inference_mnist"
docker build -t inference-client:v0.2 .