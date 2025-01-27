import os


class Configs:
    DATASET_PATH = os.environ.get(
        "PATH_DATASETS",
        "/Users/aimmo-aiy-0297/Desktop/workspace/mlops/train_mnist/data",
    )
    CHECKPOINT_PATH = os.environ.get(
        "PATH_CHECKPOINT",
        "/Users/aimmo-aiy-0297/Desktop/workspace/mlops/train_mnist/checkpoints",
    )
    RANDOM_SEED = os.environ.get("RANDOM_SEED", 10)
    OPTIMIZER = os.environ.get("OPRIMIZER", "SGD")
    LEARNING_RATE = os.environ.get("LEARNING_RATE", 0.1)
    MOMENTUM = os.environ.get("MOMENTUM", 0.9)
    WEIGHT_DECAY = os.environ.get("WEIGHT_DECAY", 1e-4)
    ACT_FN_NAME = os.environ.get("ACT_FN_NAME", "relu")
    C_HIDDEN = os.environ.get("C_HIDDEN", [16, 32, 64])
    NUM_BLOCKS = os.environ.get("NUM_BLOCKS", [3, 3, 3])
    BATCH_SISE = int(os.environ.get("BATCH_SIZE", "128"))
    EPOCHS = int(os.environ.get("EPOCHS", "10"))
    MODEL_EXPORT_PATH = os.environ.get(
        "MODEL_EXPORT_PATH",
        "/Users/aimmo-aiy-0297/Desktop/workspace/mlops/inference_mnist/resnet18",
    )
