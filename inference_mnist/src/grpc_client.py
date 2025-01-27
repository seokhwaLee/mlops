import json
import os
import random
import time
from pathlib import Path

import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image
from save_images_as_jpg import load_mnist_images, load_mnist_labels, save_images_as_jpg

url = os.environ.get("TRITON_SERVER_URL", "localhost:8001")
model_name = os.environ.get("MODEL_NAME", "resnet18")
batch_size = int(os.environ.get("BATCH_SIZE", 8))
log_sample_count = int(os.environ.get("LOG_SAMPLE_COUNT", 10))
mnist_raw_data_path = os.environ.get(
    "MNIST_RAW_DATA_PATH",
    "/Users/aimmo-aiy-0297/Desktop/workspace/mlops/train_mnist/data",
)
num_images = int(os.environ.get("NUM_IMAGES", "64"))
inference_image_dir = os.environ.get(
    "IMAGE_DIR",
    "/Users/aimmo-aiy-0297/Desktop/workspace/mlops/inference_mnist/test_datas",
)
output_dir = os.environ.get("OUTPUT_DIR", inference_image_dir)


def convert_mnist_image(raw_path, target_dir, num_images):
    mnist_image_path = f"{raw_path}/MNIST/raw/train-images-idx3-ubyte"
    mnist_label_path = f"{raw_path}/MNIST/raw/train-labels-idx1-ubyte"
    images = load_mnist_images(mnist_image_path)
    labels = load_mnist_labels(mnist_label_path)
    save_images_as_jpg(images, labels, target_dir, num_images=num_images)


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
    response = client.infer(model_name=model_name, inputs=[inputs], outputs=[outputs])
    return response.as_numpy("output")


if __name__ == "__main__":
    if not any(os.scandir(inference_image_dir)):
        convert_mnist_image(mnist_raw_data_path, inference_image_dir, num_images)

    all_results = []
    total_size = 0
    total_latency = 0.0
    output_path = f"{output_dir}/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    batches = create_batches(inference_image_dir, batch_size)
    for batch_idx, (batch_data, batch_paths) in enumerate(batches, start=1):
        start_time = time.time()
        output_data = infer_batch(batch_data)
        latency = (time.time() - start_time) * 1000
        print(f"Batch Size: {batch_data.shape[0]}, Latency: {latency:.2f} ms")

        total_size += batch_data.shape[0]
        total_latency += latency
        batch_results = {
            "batch_size": batch_data.shape[0],
            "inference_time_ms": round(latency, 2),
            "results": [],
        }
        for img_path, result in zip(batch_paths, output_data):
            predicted_class = np.argmax(result)
            batch_results["results"].append(
                {
                    "image_name": Path(img_path).name,
                    "predicted_class": str(predicted_class),  # 최대값을 가진 클래스
                    "confidence_scores": result.tolist(),
                }
            )
            all_results.append((img_path, predicted_class, result))

        batch_output_file = f"{output_path}/batch_{batch_idx}.json"
        with open(batch_output_file, "w") as f:
            json.dump(batch_results, f, indent=4)

    print(f"Total Size: {total_size} images")
    print(f"Total Latency: {total_latency:.2f} ms")
    print(f"All inference results saved to {output_path}")

    random_samples = random.sample(all_results, min(log_sample_count, len(all_results)))
    print("\nRandom Sampled Results:")
    for img_path, predicted_class, result in random_samples:
        print(f"{Path(img_path).name}: {predicted_class} | {result}")
