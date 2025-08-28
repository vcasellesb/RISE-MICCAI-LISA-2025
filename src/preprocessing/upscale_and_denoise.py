import numpy as np
from skimage.filters import unsharp_mask
from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_wavelet
)
import nibabel as nib
from bm3d import bm3d



from .brain_segmentation import segment_brain


def unsharp_masking(image: np.ndarray, radius: int, amount: int) -> np.ndarray:
    return unsharp_mask(image, radius, amount, preserve_range=True)

def apply_filter(image: np.ndarray, f, **filter_kwargs) -> np.ndarray:
    return f(image, **filter_kwargs)

def denoise_w_bm3d(image: np.ndarray, sigma_psd: float = 0.2, **bm3d_kwargs) -> np.ndarray:
    return bm3d(image, sigma_psd, **bm3d_kwargs)

def denoise_w_chambolle(image: np.ndarray, weight: float, **chambolle_kwargs) -> np.ndarray:
    return denoise_tv_chambolle(image, weight, **chambolle_kwargs)


def sharpen_and_denoise(image: np.ndarray, sharpen_kwargs: dict = None, denoise_kwargs: dict = None):
    sharpen_kwargs = sharpen_kwargs or {'radius': 3, 'amount': 1}
    image = unsharp_masking(image, **sharpen_kwargs)

    denoise_kwargs = denoise_kwargs or {'weight': 0.1}
    image = denoise_tv_chambolle(image, **denoise_kwargs)

    return image


def process_image(image_path: str, output_folder: str):

    brain_seg = segment_brain(image_path).transpose([2, 1, 0])

    image = nib.load(image_path)
    data = image.get_fdata()
    save_nifti(brain_seg, join(output_folder, 'brain_seg.nii.gz'), image.affine)

    cropped_data, brain_seg = crop_with_seg(data, brain_seg, dil=4)

    unsharped = unsharp_masking(cropped_data, radius=3, amount=1.)
    save_nifti(unsharped, join(output_folder, 'unnormalized_unsharped_radius3_amount1.nii.gz'), image.affine)

    filters = [
        denoise_tv_chambolle,
        # denoise_bilateral,
        denoise_wavelet,
        bm3d
    ]

    kwargs = [
        {'weight': 0.1},

        # {'sigma_color': 0.5,
        #  'sigma_spatial': 15},
         
        {'rescale_sigma': True},

        {'sigma_psd': 0.5}
    ]

    for i in range(len(filters)):
        this_image = filters[i](cropped_data, **kwargs[i])
        save_nifti(this_image, join(output_folder, 'unnormalized%s.nii.gz' % filters[i].__name__), image.affine)


if __name__ == "__main__":
    import os
    output_folder = 'test_upscaling'
    os.makedirs(output_folder, exist_ok=True)
    process_image('datasets/2025Task2/Low Field Images/LISA_1012_ciso.nii.gz', output_folder)