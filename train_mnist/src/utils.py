from models import MODEL_DICT


def create_model(model_name, model_hparams):
    if model_name in MODEL_DICT:
        return MODEL_DICT[model_name](**model_hparams)
    else:
        raise ValueError(f'Unknown model name "{model_name}".')
