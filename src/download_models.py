import zipfile

from HD_BET.checkpoint_download import download_file # type: ignore

from src.utils import isfile, join, remove, dirname, listdir, rmdir
from src.inference_entrypoint import TRAINED_MODELS


ZENODO_DOWNLOAD_URL = 'https://zenodo.org/records/17242234/files/trained_models.zip?download=1'

def install_model_from_zip_file(zip_file: str, target_folder: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(target_folder)


def maybe_download_parameters(url: str, target_folder: str):
    if not any(isfile(v) for v in TRAINED_MODELS.values()):
        fname = download_file(url, join(target_folder, 'tmp_trained_models.zip'))
        install_model_from_zip_file(fname, target_folder)
        remove(fname)


def recursively_remove_files_and_folders(folder_or_file: str):
    if isfile(folder_or_file):
        remove(folder_or_file)
        return

    maybe_files_or_folders = map(lambda x: join(folder_or_file, x), listdir(folder_or_file))
    for f in maybe_files_or_folders:
        recursively_remove_files_and_folders(f)

    rmdir(folder_or_file)
    return


if __name__ == "__main__":
    project_folder = dirname(dirname(__file__))
    maybe_download_parameters(ZENODO_DOWNLOAD_URL,
                              project_folder)
    # # Because I'm dumb and I made a mistake uploading the files...
    # I fixed it now uploading a new version of the folder
    # recursively_remove_files_and_folders('__MACOSX')