import os
import json
import pickle
import logging
import typing as ty
import time

import nibabel as nib
import numpy as np


remove = os.remove
listdir = os.listdir
join = os.path.join
dirname = os.path.dirname
basename = os.path.basename
exists = os.path.exists
isfile = os.path.isfile
abspath = os.path.abspath
rmdir = os.rmdir


DEFAULT_NUM_PROCESSES = 12
def get_default_num_processes() -> int:
    return min(DEFAULT_NUM_PROCESSES, os.cpu_count())

def get_default_device() -> str:
    import torch
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    return device

def maybe_mkdir(_dir: str):
    os.makedirs(_dir, exist_ok=True)

def load_json(file: str):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_pickle(file: str, mode: str = 'rb'):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a

def write_pickle(obj, file: str, mode: str = 'wb') -> None:
    with open(file, mode) as f:
        pickle.dump(obj, f)

def save_nifti(data: np.ndarray, save_path: str, affine: np.ndarray, header = None):
    nifti = nib.Nifti1Image(data, affine, header)
    nib.save(nifti, save_path)

def setup_loggers(*logger_names: str, verbosity: str, log_file: str,
                  console_verbosity: str = None, return_logger: bool = False) -> ty.Union[logging.Logger, None]:
    console_verbosity = console_verbosity or verbosity

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(filename=log_file)
    fh.setFormatter(formatter)
    fh.setLevel(verbosity)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(console_verbosity)

    for logger in logger_names:
        l = logging.getLogger(logger)

        # to avoid having a thousand fucking handlers
        if l.hasHandlers():
            continue

        l.setLevel(verbosity)
        l.addHandler(fh)
        l.addHandler(ch)

    # ugly as fuck
    if return_logger: return l


def timestampify(root: str = "") -> str:
    timestamp = time.strftime("%d%m%Y_%H%M%S")
    if len(root) and not root.endswith('_'):
        root += '_'
    return root + timestamp
