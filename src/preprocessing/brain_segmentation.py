import numpy as np

from HD_BET.hd_bet_prediction import hdbet_predict, hdbet_predict_return_array # type: ignore


def segment_brain(image: str) -> np.ndarray:
    predictor_kwargs = {
        'use_tta': False,
        'verbose': True
    }
    return hdbet_predict_return_array(image, None, predictor_kwargs)

def segment_all_brains(data_path: str, output_folder):
    hdbet_predict(data_path, output_folder, None, True, False, {'use_tta': False, 'verbose': True})


if __name__ == "__main__":
    data_path = 'datasets/2025Task2/Low Field Images'
    output_folder = 'datasets/2025Task2/Low Field Images/brainsegs'
    segment_all_brains(data_path, output_folder)
