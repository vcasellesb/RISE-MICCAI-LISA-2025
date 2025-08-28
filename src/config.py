from src.utils import abspath

from src.configs import get_default_training_config, ArchKwargs
from src.experiment_planning import plan_experiment
from src.preprocessing.preprocessing import get_preprocessing_config_from_dataset_fingerprint

TRAINING_PATH = abspath('training_data/preprocessed')
TRAINING_PATH_RAW = abspath('training_data/raw')
VALIDATION_PATH = abspath('validation_data/preprocessed')
DATASET_FINGERPRINT_PATH = abspath('training_data/raw/dataset_fingerprint.json')
LABELS = list(range(8 + 1))

def get_preprocessing_config(dataset_fingerprint_path: str = DATASET_FINGERPRINT_PATH):
    return get_preprocessing_config_from_dataset_fingerprint(dataset_fingerprint_path)

def get_configs(net: str, target_memory: int, dataset_fingerprint_path: str = DATASET_FINGERPRINT_PATH, **config_overrides):
    preprocessing_config = get_preprocessing_config(dataset_fingerprint_path)
    patch_size, arch_kwargs, batch_size, network_class = plan_experiment(preprocessing_config, net, target_memory)
    arch_kwargs = ArchKwargs(**arch_kwargs)
    training_config = get_default_training_config(patch_size=patch_size,
                                                  network_class=network_class,
                                                  batch_size=batch_size,
                                                  training_data_path=TRAINING_PATH,
                                                  validation_data_path=VALIDATION_PATH,
                                                  **config_overrides)
    return arch_kwargs, training_config, preprocessing_config