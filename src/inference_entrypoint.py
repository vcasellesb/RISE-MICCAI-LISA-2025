from .utils import timestampify, join, basename, maybe_mkdir
from .inference.predictor import Predictor
from .config import get_preprocessing_config_from_dataset_fingerprint

from HD_BET.hd_bet_prediction import hdbet_predict, hdbet_predict_sequential # type: ignore


# Hard-coded for submission
TRAINED_MODELS = {
    'unet': 'trained_models/LISA_trained_models_28_08_25/checkpoint_final.pth',
    'resunet': 'trained_models/LISA_trained_models_30_08_25/checkpoint_final.pth'
}

DATASET_FINGERPRINT_PATH = 'trained_models/dataset_fingerprint.json'

def initialize_predictor(models_path: str, verbose: bool):
    prep_config = get_preprocessing_config_from_dataset_fingerprint(DATASET_FINGERPRINT_PATH)
    predictor = Predictor.from_checkpoint_path(models_path,
                                               prep_config,
                                               allow_tqdm=False,
                                               verbose=verbose,
                                               verbose_preprocessing=verbose)
    return predictor


# I used to have this inside preprocessing in inference, but due to
# both my code and HD-BET using multiprocessing, I get the follwoing error:
# AssertionError: daemonic processes are not allowed to have children
def segment_all_brains(list_of_files: list[str], tmpdir: str,
                       num_processes_preprocessing: int, num_processes_export: int,
                       tile_step_size: float,
                       use_gaussian: bool,
                       use_tta: bool,
                       sequential: bool) -> list[str]:

    output_filenames = [join(tmpdir, basename(f).replace('.nii.gz', '_brain_seg.nii.gz')) for f in list_of_files]
    predictor_kwargs = {
        'tile_step_size': tile_step_size,
        'use_gaussian': use_gaussian,
        'use_tta': use_tta,
        'verbose': True
    }
    if sequential:
        f = hdbet_predict_sequential
    else:
        f = hdbet_predict

    f(list_of_files,
      output_filenames,
      compute_brain_extracted_image=False,
      predictor_kwargs=predictor_kwargs,
      num_processes_preprocessing=num_processes_preprocessing,
      num_processes_segmentation_export=num_processes_export)
    return output_filenames


SYNTHSR_KWARGS = {
    'force_cpu': False
}

def predict_from_files(list_of_files: list[str],
                       sequential: bool,
                       outfiles: list[str] = None,
                       tmpdir: str = None,
                       verbose: bool = True,
                       num_processes_prep: int = 1,
                       num_processes_export: int = 3,
                       nthreads_synthsr: int = 2,
                       models_path: str = None,
                       num_processes_prep_hdbet: int = 3,
                       num_processes_export_hdbet: int = 3,
                       tile_step_size_hdbet: float = 0.5,
                       use_gaussian_hdbet: bool = True,
                       use_tta_hdbet: bool = True):

    if tmpdir is None:
        tmpdir = timestampify('LISA_dockerized_inference')
    maybe_mkdir(tmpdir)

    SYNTHSR_KWARGS['nthreads'] = nthreads_synthsr

    brain_seg_paths = segment_all_brains(list_of_files, tmpdir, num_processes_prep_hdbet, num_processes_export_hdbet,
                                         tile_step_size_hdbet, use_gaussian_hdbet, use_tta_hdbet, sequential)

    predictor = initialize_predictor(TRAINED_MODELS.get(models_path, TRAINED_MODELS['unet']), verbose)

    if sequential:
        return predictor.predict_from_list_of_files_sequential(list_of_files, outfiles, brain_seg_paths,
                                                               tmpdir, SYNTHSR_KWARGS, save_probabilities=False)

    return predictor.predict_from_list_of_files(list_of_files, outfiles, brain_seg_paths, num_processes_prep, tmpdir, num_processes_export,
                                                synthsr_kwargs=SYNTHSR_KWARGS)


if __name__ == "__main__":
    from glob import glob
    files = glob('test/*.nii.gz')
    outdir = 'test-out'
    maybe_mkdir(outdir)
    outfiles = [join(outdir, basename(f)) for f in files]
    results = predict_from_files(files, outfiles)