# mlops 
[인프라 명세 및 아키텍처](https://nifty-marten-089.notion.site/MLOPS-18814179deda80808914f38dbed92f9c?pvs=4)

## 개발환경 세팅 및 실행 방법 (macOS 기준)

### 1. Minikube 설치 및 시작
Minikube를 설치하고 실행합니다. 
(프로젝트에서 사용한 minikube version: 1.35.0)
```bash
# Minikube 설치
brew install minikube

# Minikube 시작
minikube start
```

### 2. Shell Script 실행
```bash
cd script
# 2-1. Docker 이미지 빌드
sh 0-build-docker-images.sh

# 2-2. 학습 시작
# 학습 스크립트는 최초 실행 시에만 MNIST 데이터셋을 다운로드합니다.
# 학습 파라미터를 변경하고 싶은 경우 train_mnist/k8s/configmap.yaml파일을 변경한 뒤 아래 스크립트를 실행하면 됩니다.
sh 1-run-train.sh

# 2-3. Triton Server 배포
# 학습 완료 후 trison server를 배포합니다.
sh 2-deploy-triton-server.sh
# 로컬에 포트포워딩하여 triton server가 잘 떴는지 확인해볼 수 있습니다. (200 응답 확인)
kubectl port-forward svc/triton-server-service 8000:8000
curl -v localhost:8000/v2/health/ready


# 2-4. 추론(Inference) 실행
# triton server 배포 완료 후 추론을 실행합니다.
# 추론도 최초 요청 시에만 MNIST 데이터셋을 jpg로 변환하여 저장합니다.
sh 3-run-inference.sh
# 추론 이미지를 다시 만들고 싶은 경우 minikube vm에 접속하여 inference_data를 옮기거나 삭제하면 됩니다.
minikube ssh
sudo mv ~/inference_data ~/inference_data_backup
```

### 3. 결과 확인
```bash
# 3-2 로그 확인 : k9s
brew install derailed/k9s/k9s
k9s
# 리소스 검색은 ":" 를 누른 뒤 리소스명을 입력 후 enter 누르기 (ex. : > pod > enter)
# pod 로그 확인하려면 "l" 누르기
# manifest yaml확인하려면 리소스 위에서 "y" 누르기
# 이전 화면으로 돌아가려면 "esc"누르기


# 3-2 minikube vm에 접속하여 결과파일을 확인
minikube ssh
# 학습 데이터 : ~/data
# 학습모델 체크포인트 : ~/checkpoints
# 모델 : ~/models
# 추론 이미지 데이터 : ~/inference_data
# 추론 결과 데이터 : ~/inference_output_data
```
