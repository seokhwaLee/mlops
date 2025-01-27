#!/bin/bash
read -p "Enter the project root directory: " PROJECT_ROOT

# PROJECT_ROOT 유효성 검증
if [ -d "$PROJECT_ROOT" ]; then
    echo "Project root directory is set to: $PROJECT_ROOT"
else
    echo "Error: Directory '$PROJECT_ROOT' does not exist."
    exit 1
fi

cd "$PROJECT_ROOT/inference_mnist/k8s"
sed -i "" "4s/.*/  name: inference-client-${CURRENT_TIME}/" grpc-client-pod.yaml
kubectl apply -f .