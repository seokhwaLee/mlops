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
