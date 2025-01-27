import os
import random
import time

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

url = "10.97.121.159:8001"
model_name = "resnet18"
batch_size = 8
output_file = "output_results.txt"
log_sample_count = 10


def preprocess_image(image_path):
    """이미지를 전처리하여 (1, 28, 28) 형태로 반환"""
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))
    image_data = np.array(image).astype(np.float32) / 255.0
    return image_data[np.newaxis, :, :]


def create_batches(image_dir, batch_size):
    """디렉토리의 모든 이미지를 읽어 배치 단위로 반환"""
    image_paths = [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    batches = []
    current_batch = []

    for image_path in image_paths:
        current_batch.append(preprocess_image(image_path))
        if len(current_batch) == batch_size:
            batches.append((np.stack(current_batch), image_paths[:batch_size]))
            current_batch = []
            image_paths = image_paths[batch_size:]

    if current_batch:
        batches.append((np.stack(current_batch), image_paths))

    return batches


def infer_batch(batch_data):
    """TritonServer로 배치 데이터를 전송하고 결과 반환"""
    client = grpcclient.InferenceServerClient(url=url)
    inputs = grpcclient.InferInput("input", batch_data.shape, "FP32")
    inputs.set_data_from_numpy(batch_data)
    outputs = grpcclient.InferRequestedOutput("output")
    start_time = time.time()
    response = client.infer(model_name=model_name, inputs=[inputs], outputs=[outputs])
    latency = (time.time() - start_time) * 1000
    print(f"Batch Size: {batch_data.shape[0]}, Latency: {latency:.2f} ms")
    return response.as_numpy("output")


all_results = []
image_dir = "./test_datas"
batches = create_batches(image_dir, batch_size)

for batch_data, batch_paths in batches:
    output_data = infer_batch(batch_data)
    for img_path, result in zip(batch_paths, output_data):
        all_results.append((img_path, result))

with open(output_file, "w") as f:
    for img_path, result in all_results:
        f.write(f"{img_path}: {result.tolist()}\n")

print(f"All inference results saved to {output_file}")

random_samples = random.sample(all_results, min(log_sample_count, len(all_results)))
print("\nRandom Sampled Results:")
for img_path, result in random_samples:
    print(f"{img_path}: {result}")
