from glob import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation

from ..utils import join, dirname, basename, save_json

def get_id(image: str) -> str:
    return image[:-12][-4:]

def get_couple(image: str) -> tuple[str]:
    return (
        image,
        join(dirname(image), 'brainsegs', basename(image)[:-7] + '_bet.nii.gz')
    )

def process_case(image: str, dil: int = 8):
    image, brain_seg = get_couple(image)

    data: np.ndarray = nib.load(image).get_fdata()
    bseg = np.asanyarray(nib.load(brain_seg).dataobj)

    bseg = binary_dilation(bseg, iterations=dil)

    bg_values = data[~bseg]
    assert bg_values.ndim == 1
    return bg_values


def process_all_dataset():
    data_path = 'datasets/2025Task2/Low Field Images/'
    dataset = glob(join(data_path, '*.nii.gz'))
    ret = process_case(dataset[0])
    means = [ret.mean()]
    stds = [ret.std()]
    for image in dataset[1:]:
        this_ret = process_case(image)
        ret = np.concatenate((ret, this_ret))
        means.append(this_ret.mean())
        stds.append(this_ret.std())
    return {'global_mean': ret.mean(), 'global_std': ret.std(), 'means': [float(i) for i in means], 'stds': [float(i) for i in stds]}


if __name__ == "__main__":
    image = 'datasets/2025Task2/Low Field Images/LISA_0015_ciso.nii.gz'
    assert get_couple(image) == (
        image,
        'datasets/2025Task2/Low Field Images/brainsegs/LISA_0015_ciso_bet.nii.gz'
    )

    results = process_all_dataset()
    save_json(results, dirname(__file__) + '/bg_analysis.json')