# mlops

## 개발환경 세팅 및 실행 방법 (macOS 기준)

### 1. Minikube 설치 및 시작
Minikube를 설치하고 실행합니다.
```bash
# Minikube 설치
brew install minikube

# Minikube 시작
minikube start
```

### 2. Shell Script 실행
```bash
cd script
# 2-1. Docker 이미지 생성
sh 0-build-docker-images.sh

# 2-2. 학습 시작
# 학습 스크립트는 최초 실행 시에만 MNIST 데이터셋을 다운로드합니다.
# 학습 파라미터를 변경하고 싶은 경우 train_mnist/k8s/configmap.yaml파일을 변경한 뒤 아래 스크립트를 실행하면 됩니다.
sh 1-run-train.sh

# 2-3. Triton Server 배포
sh 2-deploy-triton-server.sh

# 2-4. 추론(Inference) 실행
sh 3-run-inference.sh
```