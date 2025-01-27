#!/bin/bash
read -p "Enter the project root directory: " PROJECT_ROOT

# PROJECT_ROOT 유효성 검증
if [ -d "$PROJECT_ROOT" ]; then
    echo "Project root directory is set to: $PROJECT_ROOT"
else
    echo "Error: Directory '$PROJECT_ROOT' does not exist."
    exit 1
fi

cd "$PROJECT_ROOT/triton_server/k8s"
kubectl apply -f .