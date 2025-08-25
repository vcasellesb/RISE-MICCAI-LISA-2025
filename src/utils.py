import os
import json
import pickle


remove = os.remove
listdir = os.listdir
join = os.path.join
dirname = os.path.dirname
basename = os.path.basename
exists = os.path.exists
isfile = os.path.isfile



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