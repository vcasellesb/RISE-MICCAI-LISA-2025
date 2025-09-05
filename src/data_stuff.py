from src.utils import abspath

# This file exists solely to avoid circular imports
# Out on a limb if you assuming we're friends (Assume nothing)

TRAINING_PATH_RAW = abspath('training_data/raw')

# I set both training and val data to be the same since
# I want to use all data for training
TRAINING_PATH = abspath('training_data/preprocessed')
VALIDATION_PATH = abspath('training_data/preprocessed')

DATASET_FINGERPRINT_PATH = abspath('training_data/raw/dataset_fingerprint.json')

LABELS = list(range(8 + 1))
