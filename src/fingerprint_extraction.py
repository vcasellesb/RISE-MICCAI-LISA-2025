import os
from typing import Iterable
import multiprocessing
from time import sleep
import numpy as np

from ._typing import ArrayLike
from .utils import save_json, join
from src.io import rw
from src.dataloading.dataset import generate_iterable_with_filenames

from src.preprocessing.cropping import crop_to_nonzero


# What does nnUNet extract?
# 1 - shape before and after cropping to nonzero -- not actually what the shape before cropping was,
# but actually how much cropping reduces shape.
# 2 - spacing
# 3 - some stats per case (that is, mean median min max 99.5 and 0.05 percentiles of each image)
# 4 - some (10_000) voxels that are appended to a very big list to then compute stats for the whole dataset

def collect_intensities_per_channel(
    data: np.ndarray,
    seg: np.ndarray,
    percentiles: ArrayLike,
    seed: int = 999,
    num_voxels: int = 10_000
) -> tuple[list[np.ndarray], list[dict]]:

    assert not np.isnan(data).any() and data.ndim == seg.ndim == 4
    rs = np.random.RandomState(seed)

    voxels_to_return = []
    intensity_stats = []
    fg_mask = seg[0] > 0

    for c in range(data.shape[0]):
        fg_voxels_im = data[c][fg_mask]
        num_fg_voxels = len(fg_voxels_im)

        istats = vxl = []
        if num_fg_voxels > 0:
            p = np.percentile(fg_voxels_im, percentiles)
            istats = {
                'mean': np.mean(fg_voxels_im),
                'max': fg_voxels_im.max(),
                'min': fg_voxels_im.min(),
                'percentiles': {str(percentiles[pp]): p[pp] for pp in range(len(percentiles))}
            }

            vxl = rs.choice(fg_voxels_im, size=num_voxels, replace=True)
        voxels_to_return.append(vxl)
        intensity_stats.append(istats)

    return voxels_to_return, intensity_stats

def analyze_case(
    images_paths: Iterable[str],
    seg_path: str,
    percentiles: ArrayLike,
    num_samples: int = 10_000
) -> tuple[list[np.ndarray], list[dict], tuple[int, ...], float, tuple[float, ...]]:

    data, properties = rw.read_images(images_paths)
    seg, _ = rw.read_seg(seg_path)

    spacing = properties['spacing']
    shape_before_cropping = data.shape[1:]

    # this shouldn't be here... I have done cropping using HD-BET... There should be zero cropping happening here
    # TODO: fix
    data_cropped, seg_cropped, bbox = crop_to_nonzero(data, seg)
    shape_after_cropping = data_cropped.shape[1:]
    relative_size_after_cropping = np.prod(shape_after_cropping) / np.prod(shape_before_cropping)

    some_voxels, intensity_stats_per_channel = collect_intensities_per_channel(
        data_cropped,
        seg_cropped,
        percentiles=percentiles,
        num_voxels=num_samples
    )

    return some_voxels, intensity_stats_per_channel, shape_after_cropping, relative_size_after_cropping, spacing

def run_and_save(
    input_folder: str,
    output_folder: str,
    num_processes: int,
    total_voxels_to_sample: int = 10e7,
    percentiles: ArrayLike = (50, 0.05, 99.5)
):
    _dataset = generate_iterable_with_filenames(input_folder, allow_no_seg=False, file_ending='.nii.gz')
    nvoxels_per_case = int(total_voxels_to_sample  / len(_dataset))

    r = []
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        for k in _dataset.keys():
            these_args = [(_dataset[k]['images'], _dataset[k]['seg'], percentiles, nvoxels_per_case),]
            r.append(p.starmap_async(analyze_case, these_args))

        remaining = list(range(len(_dataset)))
        # p is pretty nifti. If we kill workers they just respawn but don't do any work.
        # So we need to store the original pool of workers.
        workers = [j for j in p._pool]

        while len(remaining) > 0:
            all_alive = all([j.is_alive() for j in workers])
            if not all_alive:
                raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                    'OK jokes aside.\n'
                                    'One of your background processes is missing. This could be because of '
                                    'an error (look for an error message) or because it was killed '
                                    'by your OS due to running out of RAM. If you don\'t see '
                                    'an error message, out of RAM is likely the problem. In that case '
                                    'reducing the number of workers might help')
            done = [i for i in remaining if r[i].ready()]
            remaining = [i for i in remaining if i not in done]
            sleep(0.1)

    results = [rr.get()[0] for rr in r]

    fg_voxels_images, intensity_stats, shapes_after_cropping, relative_sizes_after_cropping, spacings = zip(*results)

    # fg_voxel_images is a list with N lists each with C arrays. N == num cases, C == num channels
    # we concatenate first for all cases then for the channels (getting a list with C arrays)
    num_channels = len(fg_voxels_images[0])
    assert all(len(fg) == num_channels for fg in fg_voxels_images[1:])
    fg_voxels_images = [np.concatenate([fg_vox[i] for fg_vox in fg_voxels_images]) for i in range(num_channels)]
    assert len(fg_voxels_images) - total_voxels_to_sample <= 1

    intensity_stats_global = {}
    for c in range(num_channels):
        _percentiles = np.percentile(fg_voxels_images[c], percentiles)
        intensity_stats_global[c] = {
            'mean': float(np.mean(fg_voxels_images[c])),
            'std': float(np.std(fg_voxels_images[c])),
            'min': float(fg_voxels_images[c].min()),
            'max': float(fg_voxels_images[c].max()),
            **{'percentile_' + str(percentiles[pp]): _percentiles[pp] for pp in range(len(percentiles))}
        }

    fingerprint = {
        'spacings': spacings,
        'median_relative_size_after_cropping': np.median(relative_sizes_after_cropping, axis=0),
        'shapes_after_cropping': shapes_after_cropping,
        'intensity_stats_full_dataset': intensity_stats_global,
        'rw_used': rw.__class__.__name__
    }

    save_json(fingerprint, join(output_folder, 'dataset_fingerprint.json'))


def entrypoint():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str)
    parser.add_argument('output_folder', type=str)
    parser.add_argument('-np', '--num_processes', type=int, required=False, default=None)

    args = parser.parse_args()
    args.num_processes = args.num_processes or os.cpu_count() // 4
    run_and_save(args.input_folder, args.output_folder, args.num_processes)

if __name__ == "__main__":

    entrypoint()