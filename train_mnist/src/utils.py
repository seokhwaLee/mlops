import os

from models import MODEL_DICT


def create_model(model_name, model_hparams):
    if model_name in MODEL_DICT:
        return MODEL_DICT[model_name](**model_hparams)
    else:
        raise ValueError(f'Unknown model name "{model_name}".')


def get_next_version(model_dir):
    try:
        version_dirs = [
            int(d)
            for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and d.isdigit()
        ]
        next_version = max(version_dirs) + 1 if version_dirs else 1
    except FileNotFoundError:
        next_version = 1

    return next_version


def get_tvt_cpu_worker_count():
    total_cores = os.cpu_count()
    if total_cores == 1:
        train_workers = 0
        val_workers = 0
        test_workers = 0
    else:
        train_workers = max(1, int(total_cores * 0.6))
        val_workers = max(1, int(total_cores * 0.3))
        test_workers = total_cores
    print(f"Total CPU cores: {total_cores}")
    print(f"Train loader workers: {train_workers}")
    print(f"Validation loader workers: {val_workers}")
    print(f"Test loader workers: {test_workers}")
    return train_workers, val_workers, test_workers
