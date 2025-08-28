from bm3d import bm3d
import numpy as np
import nibabel as nib

def save_nifti(array, path, affine):
    nib.save(nib.Nifti1Image(array, affine), path)

profiles = ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb')

high_noisy_data = 'datasets/2025Task2/Low Field Images/LISA_1004_ciso.nii.gz'
high_noisy_data = nib.load(high_noisy_data)
noisy_data = high_noisy_data.get_fdata()

def perform_bm3d_slice_wise(data: np.ndarray, sigma_psd, profile) -> np.ndarray:
    z_dim = data.shape[-1]
    out = np.zeros_like(data)
    for z in range(z_dim):
        print('Working on slice %i of %i' % (z, z_dim))
        out[..., z] = bm3d(data[..., z], sigma_psd=sigma_psd, profile=profile)    
    return out

for profile in profiles:
    denoised_with_profile = perform_bm3d_slice_wise(noisy_data, 0.1, profile)
    save_nifti(denoised_with_profile, 'LISA_1004_ciso_bm3d_%s.nii.gz' % profile, high_noisy_data.affine)