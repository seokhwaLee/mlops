import os

import pytorch_lightning as pl
import torch
import torch.utils.data as data
from configs import Configs
from torchvision import transforms
from torchvision.datasets import MNIST
from utils import get_tvt_cpu_worker_count

DATA_MEANS = 0.13066047430038452
DATA_STD = 0.30810782313346863

test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(DATA_MEANS, DATA_STD)]
)
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEANS, DATA_STD),
    ]
)

download = False if os.path.exists(f"{Configs.DATASET_PATH}/MNIST") else True

train_dataset = MNIST(
    root=Configs.DATASET_PATH, train=True, transform=train_transform, download=download
)
val_dataset = MNIST(
    root=Configs.DATASET_PATH, train=True, transform=test_transform, download=download
)
pl.seed_everything(42)
train_set, _ = torch.utils.data.random_split(train_dataset, [54000, 6000])
pl.seed_everything(42)
_, val_set = torch.utils.data.random_split(val_dataset, [54000, 6000])

test_set = MNIST(
    root=Configs.DATASET_PATH, train=False, transform=test_transform, download=download
)

train_workers, val_workers, test_workers = get_tvt_cpu_worker_count()

train_loader = data.DataLoader(
    train_set,
    batch_size=Configs.BATCH_SISE,
    shuffle=True,
    drop_last=True,
    num_workers=train_workers,
)
val_loader = data.DataLoader(
    val_set,
    batch_size=Configs.BATCH_SISE,
    shuffle=False,
    drop_last=False,
    num_workers=val_workers,
)
test_loader = data.DataLoader(
    test_set,
    batch_size=Configs.BATCH_SISE,
    shuffle=False,
    drop_last=False,
    num_workers=test_workers,
)
