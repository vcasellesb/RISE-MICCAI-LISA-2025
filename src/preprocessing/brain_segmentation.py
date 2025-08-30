from collections.abc import Callable
import numpy as np

from HD_BET.hd_bet_prediction import hdbet_predict, hdbet_predict_return_array, get_hdbet_predictor # type: ignore


def segment_brain(image: str) -> np.ndarray:
    predictor_kwargs = {
        'use_tta': False,
        'verbose': True
    }
    return hdbet_predict_return_array(image, None, predictor_kwargs)

def segment_brain_initialize_predictor_once() -> Callable[[str], np.ndarray]:
    predictor_kwargs = {
        'use_tta': False,
        'verbose': True
    }
    predictor = get_hdbet_predictor(**predictor_kwargs)

    def wrapped(image: str) -> np.ndarray:
        return hdbet_predict_return_array(image, predictor, None)

    return wrapped

def segment_all_brains(data_path: str, output_folder):
    hdbet_predict(data_path, output_folder, None, True, False, {'use_tta': False, 'verbose': True})


if __name__ == "__main__":
    import glob
    data_path = 'datasets/2025Task2/Low Field Images'
    output_folder = 'datasets/2025Task2/Low Field Images/brainsegs'
    # segment_all_brains(data_path, output_folder)


    lowfield_scans = glob.glob(data_path + '/*.nii.gz')
    segmentor = segment_brain_initialize_predictor_once()
    for l in lowfield_scans:
        x = segmentor(l)
        print(x.shape)
