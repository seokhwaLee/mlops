# 베이스 이미지 설정
FROM python:3.10.16-slim

WORKDIR /workspace/src

COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir -r /workspace/requirements.txt

COPY src /workspace/src

CMD ["python", "train.py"]
