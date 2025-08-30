import typing as ty
import numpy as np
import SimpleITK as sitk

from src._typing import ArrayLike, ToIterableInt
from src.utils import join, basename
from src.io import rw


from .resampling import (
    compute_new_shape,
)
from .cropping_ciso import crop_with_seg
from .upscale_and_denoise import denoise_w_chambolle
from .synthsr import run_synthsr_on_lowfield_scan



def load_image(image: str, target_dtype) -> tuple[np.ndarray, dict[str, dict]]:
    """
    This is a compatibility layer with src.io.rw -- just not to be handling rw returning data as 4D arrays...
    I.E., This loads as 3D arrays!
    """
    sitk_image = sitk.ReadImage(image)
    data = sitk.GetArrayFromImage(sitk_image).astype(target_dtype)
    spacing_original = sitk_image.GetSpacing()
    properties = {
        'sitk_stuff': {
            'spacing': spacing_original,
            'origin': sitk_image.GetOrigin(),
            'direction': sitk_image.GetDirection()
        },
        'spacing': list(np.abs(spacing_original[::-1]))
    }
    return data, properties


def save_image(
    data: np.ndarray,
    output_fname: str,
    properties: dict,
    target_dtype
) -> None:
    assert data.ndim == 3
    itk_image = sitk.GetImageFromArray(data.astype(target_dtype, copy=False))
    itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
    itk_image.SetOrigin(properties['sitk_stuff']['origin'])
    itk_image.SetDirection(properties['sitk_stuff']['direction'])

    sitk.WriteImage(itk_image, output_fname, useCompression=True)


def _preprocess_case_inference(
    data: np.ndarray,
    properties: dict,
    normalizers_per_channel: ty.Iterable[ty.Callable[[np.ndarray, bool, ty.Optional[np.ndarray]], np.ndarray]],
    target_spacing: ArrayLike,
    transpose_forward: ToIterableInt,
    resampling_data_function: ty.Callable,
    verbose: bool = True
) -> tuple[np.ndarray, dict]:
    """
    This function requires the target spacing that will be used throughout the model, as well as any
    transposition that should be required.

    Order of operations:
        1. Transpose (to bring lowres axis to first dimension) - data + seg (if available) + spacing.
        2. Crop to nonzero (generates an OR mask where any of the data channels are non_zero).
        3. Normalization - for MRI, this should be ZScore.
        4. Resampling to target spacing.
        5. If the data has a segmentation - i.e., it is a training case - foreground locations are extracted,
            for later oversampling of foreground regions during training data loading.
    """

    assert len(data) == len(normalizers_per_channel), 'One would expect the same number of channels as normalization functions per channel...'

    data = data.transpose([0, *[i + 1 for i in transpose_forward]])

    original_spacing = [properties['spacing'][i] for i in transpose_forward]
    new_shape = compute_new_shape(properties['shape_after_cropping'], original_spacing, target_spacing)

    for c in range(data.shape[0]):
        if verbose:
            print('Normalizing channel %i with scheme: %s' % (c, normalizers_per_channel[c].func.__name__))
        data[c] = normalizers_per_channel[c](data[c])

    if verbose:
        print('Resampling from old shape %s to new shape %s (target spacing - %s).' % (properties["shape_after_cropping"], new_shape, target_spacing))

    data = resampling_data_function(data, new_shape, original_spacing, target_spacing)

    properties['shape_after_resampling'] = data.shape[1:]

    return data, properties


def preprocess_case(lowfield_scan: str,
                    preprocessing_kwargs: dict,
                    tmpdir: str,
                    brain_seg_path: str):

    seg, _ = load_image(brain_seg_path, target_dtype=np.uint8)

    # rw returns as 4D. Let's denoise rememebering to go back and forth to it
    data, properties = load_image(lowfield_scan, np.float32)
    properties['shape_before_cropping'] = data.shape

    data, seg, _, bbox = crop_with_seg(data, seg)
    properties['shape_after_cropping'] = data.shape
    properties['bbox_used_for_cropping'] = bbox

    data = denoise_w_chambolle(data, weight=None)

    tmp_lowfield_scan = join(tmpdir, basename(lowfield_scan))
    save_image(data, tmp_lowfield_scan, properties, np.float32)

    tmp_sr_scan = join(tmpdir, basename(lowfield_scan).replace('.nii.gz', '_SR.nii.gz'))

    run_synthsr_on_lowfield_scan(tmp_lowfield_scan, tmp_sr_scan, nthreads=2)

    # now prepare data for normalization etc
    data, _ = rw.read_images((tmp_lowfield_scan, tmp_sr_scan))

    return _preprocess_case_inference(data, properties, **preprocessing_kwargs)