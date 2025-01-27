#!/bin/bash
read -p "Enter the project root directory: " PROJECT_ROOT

# PROJECT_ROOT 유효성 검증
if [ -d "$PROJECT_ROOT" ]; then
    echo "Project root directory is set to: $PROJECT_ROOT"
else
    echo "Error: Directory '$PROJECT_ROOT' does not exist."
    exit 1
fi

# train job 배포
cd "$PROJECT_ROOT/train_mnist/k8s"
kubectl delete -f train-job.yaml
kubectl apply -f .