from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from configs import Configs
from utils import create_model, get_next_version


class MNISTTrainer(pl.LightningModule):
    def __init__(self, model_name, model_hparams, optimizer_name, optimizer_hparams):
        """MNISTTrainer.

        Args:
            model_name: Name of the model/CNN to run. Used for creating the model (see function below)
            model_hparams: Hyperparameters for the model, as dictionary.
            optimizer_name: Name of the optimizer to use. Currently supported: Adam, SGD
            optimizer_hparams: Hyperparameters for the optimizer, as dictionary. This includes learning rate, weight decay, etc.

        """
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name, model_hparams)
        self.loss_module = nn.CrossEntropyLoss()
        self.example_input_array = torch.zeros((1, 1, 28, 28), dtype=torch.float32)

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log("train_acc", acc, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True)
        print(
            f"[Training] Step {batch_idx}: Loss={loss.item():.4f}, Acc={acc.item():.4f}"
        )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("val_acc", acc, prog_bar=False)
        print(f"[Validation] Step {batch_idx}: Acc={acc.item():.4f}")

    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("test_acc", acc, prog_bar=False)
        print(f"[Test] Step {batch_idx}: Acc={acc.item():.4f}")

    def save_model(self):
        version = get_next_version(Configs.MODEL_EXPORT_PATH)
        model_path = f"{Configs.MODEL_EXPORT_PATH}/{version}"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.model.save_model(f"{model_path}/model.pt")
        if version == 1:
            with open(f"{Configs.MODEL_EXPORT_PATH}/config.pbtxt", "w") as f:
                f.write("""
name: "resnet18"
platform: "pytorch_libtorch"
max_batch_size: 8
input [
{
    name: "input"
    data_type: TYPE_FP32
    dims: [1, 28, 28]
}
]
output [
{
    name: "output"
    data_type: TYPE_FP32
    dims: [10]
}
]
    """)
        return model_path
