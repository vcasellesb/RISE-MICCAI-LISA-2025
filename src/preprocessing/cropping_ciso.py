from collections.abc import Callable
import numpy as np
from skimage.measure import label

from .cropping import get_bbox_from_mask, bounding_box_to_slice, binary_dilation, binary_fill_holes


DEFAULT_DIL = 5
def crop_with_seg(
    data: np.ndarray,
    seg: np.ndarray,
    dil: int = DEFAULT_DIL
) -> tuple:

    if dil > 0:
        seg = binary_dilation(seg, iterations=dil)

    bbox = get_bbox_from_mask(seg)
    slicer = bounding_box_to_slice(bbox)
    seg = seg[slicer]
    data = data[slicer]

    return data, seg, slicer, bbox

def crop_to_nonzero_ciso(
    data: np.ndarray,
    nonzero_label = -1,
    ciso_thr: float = 1.,
    dil: int = 0
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """We work with 3d data."""
    nonzero_mask = create_nonzero_mask_ciso(data, ciso_thr, dil)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer]

    data = data[slicer]
    seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))

    return data, seg, bbox

def create_nonzero_mask_ciso(data: np.ndarray, ciso_thr: float = 1., dil: int = 0) -> np.ndarray:
    assert data.ndim == 3
    nonzero_mask = data > ciso_thr
    nonzero_mask = binary_fill_holes(nonzero_mask)
    nonzero_mask = remove_all_but_largest_component_from_segmentation(nonzero_mask)
    if dil > 0:
        nonzero_mask = binary_dilation(nonzero_mask, iterations=dil)
    return nonzero_mask


def label_with_component_sizes(binary_image: np.ndarray, connectivity: int = None) -> tuple[np.ndarray, dict]:
    if not binary_image.dtype == bool:
        print('Warning: it would be way faster if your binary image had dtype bool')
    labeled_image, num_components = label(binary_image, return_num=True, connectivity=connectivity)
    component_sizes = {i + 1: j for i, j in enumerate(np.bincount(labeled_image.ravel())[1:])}
    return labeled_image, component_sizes


def generic_filter_components(
    binary_image: np.ndarray,
    filter_fn: Callable[[list[int], list[int]], list[int]],
    connectivity: int = None
):
    """
    filter_fn MUST return the component ids that should be KEPT!
    filter_fn will be called as: filter_fn(component_ids, component_sizes) and is expected to return a List of int

    returns a binary array that is True where the filtered components are
    """
    labeled_image, component_sizes = label_with_component_sizes(binary_image, connectivity)
    component_ids = list(component_sizes.keys())
    component_sizes = list(component_sizes.values())
    keep = filter_fn(component_ids, component_sizes)
    return np.isin(labeled_image.ravel(), keep).reshape(labeled_image.shape)


def remove_all_but_largest_component(binary_image: np.ndarray, connectivity: int = None) -> np.ndarray:
    """
    Removes all but the largest component in binary_image. Replaces pixels that don't belong to it with background_label
    """
    filter_fn = lambda x, y: [i for i, j in zip(x, y) if j == max(y)]
    return generic_filter_components(binary_image, filter_fn, connectivity)


def remove_all_but_largest_component_from_segmentation(seg: np.ndarray, background_label: int = 0) -> np.ndarray:
    mask = seg.astype(bool, copy=True)
    mask = remove_all_but_largest_component(mask)
    seg[~mask] = background_label
    return seg

if __name__ == "__main__":
    import nibabel as nib
    im = nib.load('datasets/2025Task2/Low Field Images/LISA_1012_ciso.nii.gz')
    brainseg = create_nonzero_mask_ciso(im.get_fdata())
    label_ = nib.Nifti1Image(brainseg.astype(np.uint8), im.affine)
    nib.save(label_, 'brainseg_rough.nii.gz')