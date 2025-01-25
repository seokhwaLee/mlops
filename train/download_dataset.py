import os

from torchvision import transforms
from torchvision.datasets import MNIST

DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
mnist_dataset = MNIST(root=DATASET_PATH, train=True, download=True)
DATA_MEANS = (mnist_dataset.data / 255.0).mean(axis=(0, 1, 2))
DATA_STD = (mnist_dataset.data / 255.0).std(axis=(0, 1, 2))

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)]
)
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD),
    ]
)
# Loading the training dataset. We need to split it into a training and validation part
# We need to do a little trick because the validation set should not use the augmentation.
train_dataset = MNIST(
    root=DATASET_PATH, train=True, transform=train_transform, download=True
)
val_dataset = MNIST(
    root=DATASET_PATH, train=True, transform=test_transform, download=True
)

# Loading the test set
test_set = MNIST(
    root=DATASET_PATH, train=False, transform=test_transform, download=True
)
