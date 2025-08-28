import numpy as np
import nibabel as nib
from skimage.restoration import estimate_sigma


from .utils import join, maybe_mkdir, exists, save_nifti

from .preprocessing.brain_segmentation import segment_brain
from .preprocessing.cropping_ciso import crop_with_seg
from .preprocessing.upscale_and_denoise import denoise_w_chambolle, sharpen_and_denoise
from .preprocessing.synthsr import run_synthsr_on_lowfield_scan

from .dataloading.dataset import get_identifiers


RAW_DATASET_PATH = 'datasets/2025Task2/'

# Proposta actual:
# 1. Hd-Bet
# 2. Cropping amb brainseg
# 3. Bm3d
# 4. SynthSR


def get_raw_case_dict(identifier: str) -> dict[str, str]:
    this_case = {
        'LF': join(RAW_DATASET_PATH, 'Low Field Images', 'LISA_%s_ciso.nii.gz' % identifier),
        'HIPP': join(RAW_DATASET_PATH, 'Subtask 2a - Hippocampus Segmentations', 'LISA_%s_HF_hipp.nii.gz' % identifier),
        'BAGA': join(RAW_DATASET_PATH, 'Subtask 2b - Basal Ganglia Segmentations', 'LISA_%s_HF_baga.nii.gz' % identifier),
        'VENT': join(RAW_DATASET_PATH, 'Extra Segmentations', 'Ventricle', 'LISA_%s_vent.nii.gz' % identifier)
    }
    return this_case


def prepare_segmentation(hipp: str, baga: str, vent: str) -> np.ndarray:
    hipp = nib.load(hipp)
    seg: np.ndarray = np.asanyarray(hipp.dataobj)
    assert set(np.unique(seg.ravel())) == {0, 1, 2}, set(np.unique(seg.ravel()))

    vent_seg = np.asanyarray(nib.load(vent).dataobj)

    baga_seg: np.ndarray = np.asanyarray(nib.load(baga).dataobj)
    assert (
        baga_seg.shape == seg.shape == vent_seg.shape and
        set(np.unique(baga_seg.ravel())) == {0, 5, 6, 7, 8} and
        set(np.unique(vent_seg.ravel())) == {0, 3, 4}
    ), baga

    intersection = seg * baga_seg
    if intersection.any():
        assert baga.endswith('LISA_0033_HF_baga.nii.gz')
        baga_seg = np.where(intersection, 0, baga_seg)
    seg = seg + baga_seg
    del baga_seg

    intersection = seg * vent_seg
    if intersection.any():
        print('Intersection between vent and hipp|baga')
        vent_seg = np.where(intersection, 0, vent_seg)

    return vent_seg + seg

def get_weight(data: np.ndarray) -> float:
    sigma_hat = estimate_sigma(data)
    return np.round(sigma_hat, 2) * 10 + 0.1

WEIGHT_CHAMBOLLE = 0.2
def process_lowfield_image(
    image: str,
    gt: np.ndarray,
    lowfield_out: str,
    sr_out: str,
    gt_out: str
) -> None:

    # HD-BET uses sitk so we need to transpose data
    brain_segmentation = segment_brain(image).transpose([2, 1, 0])

    im: nib.Nifti1Image = nib.load(image)
    data = im.get_fdata()
    cropped_data, _, slicer = crop_with_seg(data, brain_segmentation, dil=5)

    weight = get_weight(cropped_data)

    denoised_data = denoise_w_chambolle(cropped_data, **{'weight': weight})

    save_nifti(denoised_data, lowfield_out, im.affine)

    run_synthsr_on_lowfield_scan(lowfield_out, sr_out, nthreads=2)

    gt = gt[slicer]
    save_nifti(gt.astype(np.uint8), gt_out, im.affine)


def prepare_output_filenames(identifier, output_folder):
    joiner = lambda x: join(output_folder, x % identifier)
    _t = map(
        joiner,
        ('LISA_%s_LF.nii.gz',
        'LISA_%s_SR.nii.gz',
        'LISA_%s_GT.nii.gz')
    )
    return _t


def prepare_raw_training_data(
    output_folder: str
) -> None:
    maybe_mkdir(output_folder)
    identifiers = get_identifiers(join(RAW_DATASET_PATH, 'Low Field Images'))
    for identifier in identifiers:
        case_dict = get_raw_case_dict(identifier)
        lowfield_out, sr_out, seg_out = prepare_output_filenames(identifier, output_folder)
        if all(map(exists, (lowfield_out, sr_out, seg_out))):
            continue
        lowfield_in = case_dict.pop('LF')
        gt = prepare_segmentation(**{k.lower(): v for k, v in case_dict.items()})
        process_lowfield_image(lowfield_in, gt, lowfield_out, sr_out, seg_out)



def mylilcheck():
    identifiers = get_identifiers(join(RAW_DATASET_PATH, 'Low Field Images'))
    i = identifiers.pop(0)
    case_dict = get_raw_case_dict(i)
    hipp_seg = np.asanyarray(nib.load(case_dict['HIPP']).dataobj)
    baga_seg = np.asanyarray(nib.load(case_dict['BAGA']).dataobj)
    set_hipp = set(np.unique(hipp_seg.ravel()))
    set_baga = set(np.unique(baga_seg.ravel()))

    for i in identifiers:
        case_dict = get_raw_case_dict(i)
        hipp_seg = np.asanyarray(nib.load(case_dict['HIPP']).dataobj)
        baga_seg = np.asanyarray(nib.load(case_dict['BAGA']).dataobj)
        set_hipp |= set(np.unique(hipp_seg.ravel()))
        set_baga |= set(np.unique(baga_seg.ravel()))
    

    assert set_hipp == {0, 1, 2}, set_hipp
    assert set_baga == {0, 5, 6, 7, 8}, set_baga


if __name__ == "__main__":
    lowfield, sr, seg = prepare_output_filenames('0001', 'liltest')
    assert (
        (lowfield, sr, seg) ==
        (
            'liltest/LISA_0001_LF.nii.gz',
            'liltest/LISA_0001_SR.nii.gz',
            'liltest/LISA_0001_GT.nii.gz'
        )
    )

    from .config import TRAINING_PATH_RAW
    prepare_raw_training_data(
        TRAINING_PATH_RAW
    )
    # mylilcheck()