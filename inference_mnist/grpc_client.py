import numpy as np
import tritonclient.grpc as grpcclient
from PIL import Image

image = Image.open("./test_datas/mnist_2_label_4.jpg").convert("L")
image = image.resize((28, 28))
image_data = np.array(image).astype(np.float32) / 255.0
image_data = image_data[np.newaxis, np.newaxis, :, :]

url = "10.97.121.159:8001"

client = grpcclient.InferenceServerClient(url=url)

model_name = "resnet18"
model_version = ""

inputs = grpcclient.InferInput("input", image_data.shape, "FP32")
inputs.set_data_from_numpy(image_data)

outputs = grpcclient.InferRequestedOutput("output")

response = client.infer(model_name=model_name, inputs=[inputs], outputs=[outputs])

output_data = response.as_numpy("output")
print("Inference Result:", output_data)
