import numpy as np
import requests
from PIL import Image

image = Image.open("./test_datas/mnist_2_label_4.jpg").convert("L")
image = image.resize((28, 28))
image_data = np.array(image).astype(np.float32) / 255.0
image_data = image_data[np.newaxis, np.newaxis, :, :]

data = {
    "inputs": [
        {
            "name": "input",
            "shape": image_data.shape,
            "datatype": "FP32",
            "data": image_data.flatten().tolist(),
        }
    ]
}

response = requests.post(
    "http://10.97.121.159:8000/v2/models/resnet18/infer", json=data
)
print(response.json())
