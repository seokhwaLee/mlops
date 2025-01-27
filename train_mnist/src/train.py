import os

import pytorch_lightning as pl
import torch
from configs import Configs
from datasets.mnist_loader import test_loader, train_loader, val_loader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from trainers.mnist_trainer import MNISTTrainer

pl.seed_everything(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.device("cpu")


def train_model(model_name, save_name=None, **kwargs):
    """Train model.

    Args:
        model_name: Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional): If specified, this name will be used for creating the checkpoint and logging directory.

    """
    if save_name is None:
        save_name = model_name

    trainer = pl.Trainer(
        default_root_dir=os.path.join(Configs.CHECKPOINT_PATH, save_name),
        accelerator="auto",
        devices=1,
        max_epochs=Configs.EPOCHS,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pretrained_filename = os.path.join(Configs.CHECKPOINT_PATH, save_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = MNISTTrainer.load_from_checkpoint(pretrained_filename)
    else:
        print(f"Not Found pretrained model at {pretrained_filename}, pass...")
        pl.seed_everything(Configs.RANDOM_SEED)
        model = MNISTTrainer(model_name=model_name, **kwargs)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    model = MNISTTrainer.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"]}
    print(f"accuracy - {result}")
    return model, result


if __name__ == "__main__":
    resnet_model, resnet_results = train_model(
        model_name="ResNet",
        model_hparams={
            "num_classes": 10,
            "c_hidden": Configs.C_HIDDEN,
            "num_blocks": Configs.NUM_BLOCKS,
            "act_fn_name": Configs.ACT_FN_NAME,
        },
        optimizer_name="SGD",
        optimizer_hparams={
            "lr": Configs.LEARNING_RATE,
            "momentum": Configs.MOMENTUM,
            "weight_decay": Configs.WEIGHT_DECAY,
        },
    )

    model_path = resnet_model.save_model()
    print(f"model saved : {model_path}")
