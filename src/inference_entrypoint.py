from .utils import timestampify, join, basename, maybe_mkdir
from .inference.predictor import Predictor
from .config import DATASET_FINGERPRINT_PATH, get_preprocessing_config_from_dataset_fingerprint

from HD_BET.hd_bet_prediction import hdbet_predict # type: ignore


TRAINED_MODELS_CHECKPOINT_PATH = 'trained_models/LISA_trained_models_28_08_25/checkpoint_final.pth'

def initialize_predictor():
    prep_config = get_preprocessing_config_from_dataset_fingerprint(DATASET_FINGERPRINT_PATH)
    predictor = Predictor.from_checkpoint_path(TRAINED_MODELS_CHECKPOINT_PATH,
                                               prep_config,
                                               allow_tqdm=False)
    return predictor


# I used to have this inside preprocessing in inference, but due to
# both my code and HD-BET using multiprocessing, I get the follwoing error:
# AssertionError: daemonic processes are not allowed to have children
def segment_all_brains(list_of_files: list[str], tmpdir: str) -> list[str]:
    output_filenames = [join(tmpdir, basename(f).replace('.nii.gz', '_brain_seg.nii.gz')) for f in list_of_files]
    predictor_kwargs = {
        'tile_step_size': 1,
        'use_gaussian': False,
        'use_tta': False,
        'verbose': True
    }
    hdbet_predict(list_of_files,
                  output_filenames,
                  compute_brain_extracted_image=False,
                  predictor_kwargs=predictor_kwargs)
    return output_filenames


def predict_from_files(list_of_files: list[str],
                       outfiles: list[str] = None,
                       tmpdir: str = None,
                       verbose: bool = True,
                       num_processes_prep: int = 3,
                       num_processes_export: int = 3):
    if tmpdir is None:
        tmpdir = timestampify('LISA_dockerized_inference')
    brain_seg_paths = segment_all_brains(list_of_files, tmpdir)
    predictor = initialize_predictor()
    return predictor.predict_from_list_of_files(list_of_files, outfiles, brain_seg_paths, verbose, num_processes_prep, tmpdir, num_processes_export)


if __name__ == "__main__":
    from glob import glob
    files = glob('test/*.nii.gz')
    outdir = 'test-out'
    maybe_mkdir(outdir)
    outfiles = [join(outdir, basename(f)) for f in files]
    results = predict_from_files(files, outfiles)