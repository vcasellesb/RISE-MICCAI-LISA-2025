from src.utils import abspath

# This file exists solely to avoid circular imports
# Out on a limb if you assuming we're friends (Assume nothing)

TRAINING_PATH = abspath('training_data/preprocessed')
TRAINING_PATH_RAW = abspath('training_data/raw')
VALIDATION_PATH = abspath('training_data/preprocessed')

DATASET_FINGERPRINT_PATH = abspath('training_data/raw/dataset_fingerprint.json')

LABELS = list(range(8 + 1))
