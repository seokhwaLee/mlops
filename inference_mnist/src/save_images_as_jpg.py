import os
import struct

import numpy as np
from PIL import Image


def load_mnist_images(file_path):
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    return images


def load_mnist_labels(file_path):
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def save_images_as_jpg(images, labels, output_dir, num_images=10):
    """
    Args:
        images (np.ndarray): (N, 28, 28) 크기의 이미지 데이터
        labels (np.ndarray): 각 이미지의 레이블
        output_dir (str): 저장 디렉토리 경로
        num_images (int): 저장할 이미지 개수
    """
    os.makedirs(output_dir, exist_ok=True)
    for i in range(min(num_images, len(images))):
        image = Image.fromarray(images[i])
        label = labels[i]
        file_path = os.path.join(output_dir, f"mnist_{i}_label_{label}.jpg")
        image.save(file_path)
        print(f"Saved: {file_path}")


if __name__ == "__main__":
    MNIST_IMAGE_PATH = "/Users/aimmo-aiy-0297/Desktop/workspace/mlops/train_mnist/data/MNIST/raw/train-images-idx3-ubyte"
    MNIST_LABEL_PATH = "/Users/aimmo-aiy-0297/Desktop/workspace/mlops/train_mnist/data/MNIST/raw/train-labels-idx1-ubyte"
    OUTPUT_DIR = (
        "/Users/aimmo-aiy-0297/Desktop/workspace/mlops/inference_mnist/test_datas"
    )
    NUM_IMAGES = 10

    images = load_mnist_images(MNIST_IMAGE_PATH)
    labels = load_mnist_labels(MNIST_LABEL_PATH)

    save_images_as_jpg(images, labels, OUTPUT_DIR, num_images=NUM_IMAGES)
