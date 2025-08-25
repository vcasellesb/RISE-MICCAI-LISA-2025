import numpy as np
import nibabel as nib


from .utils import join, maybe_mkdir, exists

from .preprocessing.brain_segmentation import segment_brain
from .preprocessing.cropping_ciso import crop_with_seg
from .preprocessing.upscale_and_denoise import denoise_w_bm3d
from .preprocessing.synthsr import run_synthsr_on_lowfield_scan

from .dataloading.dataset import get_identifiers

from .paths import TRAINING_DATA


RAW_DATASET_PATH = 'datasets/2025Task2/'

# Proposta actual:
# 1. Hd-Bet
# 2. Cropping amb brainseg
# 3. Bm3d
# 4. SynthSR


def save_nifti(data: np.ndarray, save_path: str, affine: np.ndarray, header = None):
    nifti = nib.Nifti1Image(data, affine, header)
    nib.save(nifti, save_path)


def get_raw_case_dict(identifier: str) -> dict[str, str]:
    this_case = {
        'LF': join(RAW_DATASET_PATH, 'Low Field Images', 'LISA_%s_ciso.nii.gz' % identifier),
        'HIPP': join(RAW_DATASET_PATH, 'Subtask 2a - Hippocampus Segmentations', 'LISA_%s_HF_hipp.nii.gz' % identifier),
        'BAGA': join(RAW_DATASET_PATH, 'Subtask 2b - Basal Ganglia Segmentations', 'LISA_%s_HF_baga.nii.gz' % identifier)
    }
    return this_case


def prepare_segmentation(hipp: str, baga: str) -> np.ndarray:
    hipp = nib.load(hipp)
    seg: np.ndarray = np.asanyarray(hipp.dataobj)
    assert set(np.unique(seg.ravel())) == {0, 1, 2}, set(np.unique(seg.ravel()))

    baga_seg: np.ndarray = np.asanyarray(nib.load(baga).dataobj)
    assert (
        baga_seg.shape == seg.shape and
        set(np.unique(baga_seg.ravel())) == {0, 5, 6, 7, 8}
    ), baga

    # we need to go from [5, 6, 7, 8] to [3, 4, 5, 6]
    # I try to be fancy w it
    mask = baga_seg > 0
    baga_seg[mask] = baga_seg[mask] - 2

    # I do have OCD tho
    assert (
        baga_seg.shape == seg.shape and
        set(np.unique(baga_seg.ravel())) == {0, 3, 4, 5, 6}
    ), baga

    if (seg * baga_seg).any():
        assert baga.endswith('LISA_0033_HF_baga.nii.gz')
        baga_seg = np.where(baga_seg * seg, 0, baga_seg)

    assert not (seg * baga_seg).any(), baga

    return baga_seg + seg


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

    denoised_data = denoise_w_bm3d(cropped_data)

    save_nifti(denoised_data, lowfield_out, im.affine)

    run_synthsr_on_lowfield_scan(lowfield_out, sr_out, nthreads=2)

    gt = gt[slicer]
    save_nifti(gt, gt_out, im.affine)


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

    prepare_raw_training_data(
        join(TRAINING_DATA, 'raw')
    )
    # mylilcheck()