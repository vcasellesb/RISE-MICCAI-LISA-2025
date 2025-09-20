import typing as ty
from time import sleep
import multiprocessing
from functools import partial
import numpy as np

from src._typing import ArrayLike, ToIterableInt
from src.utils import join, maybe_mkdir, remove, dirname, get_default_num_processes
from src.io import rw
from src.dataloading.dataset import Dataset, generate_iterable_with_filenames

from src.data_stuff import LABELS

from src.configs.preprocessing import get_preprocessing_config_from_dataset_fingerprint

from .normalization import zscore_norm
from .cropping import crop_to_nonzero
from .resampling import (
    compute_new_shape,
    resample_data_or_seg_to_shape
)


def _preprocess_case(
    data: np.ndarray,
    properties: dict,
    normalizers_per_channel: ty.Iterable[ty.Callable[[np.ndarray, bool, ty.Optional[np.ndarray]], np.ndarray]],
    target_spacing: ArrayLike,
    transpose_forward: ToIterableInt,
    resampling_function: ty.Callable,
    resampling_seg_function: ty.Callable,
    seg: np.ndarray = None,
    verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, dict]:
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
    has_seg = seg is not None

    data = data.transpose([0, *[i + 1 for i in transpose_forward]])
    if has_seg:
        seg = seg.transpose([0, *[i + 1 for i in transpose_forward]])
    original_spacing = [properties['spacing'][i] for i in transpose_forward]

    properties['shape_before_cropping'] = data.shape[1:]
    data, seg, bbox = crop_to_nonzero(data, seg, nonzero_label=-1)
    properties['shape_after_cropping'] = data.shape[1:]
    properties['bbox_used_for_cropping'] = bbox

    new_shape = compute_new_shape(properties['shape_after_cropping'], original_spacing, target_spacing)

    for c in range(data.shape[0]):
        if verbose:
            print('Normalizing channel %i with scheme: %s' % (c, normalizers_per_channel[c].func.__name__))
        data[c] = normalizers_per_channel[c](data[c])

    if verbose:
        print('Resampling from old shape %s to new shape %s (target spacing - %s).' % (properties["shape_after_cropping"], new_shape, target_spacing))

    data = resampling_function(data, new_shape, original_spacing, target_spacing)
    seg = resampling_seg_function(seg, new_shape, original_spacing, target_spacing)

    properties['shape_after_resampling'] = data.shape[1:]

    seg = seg.astype(np.int8 if seg.max() <= 255. else np.int16)

    if has_seg:
        properties['class_locations'] = sample_foreground_locations(seg, classes=LABELS, verbose=verbose)

    return data, seg, properties


def sample_foreground_locations(seg: np.ndarray,
                                classes: int | ty.Iterable[int],
                                num_samples: int = 10_000,
                                seed: int = 999,
                                verbose: bool = True,
                                min_percentage_covered: float = .01) -> dict:

    rndst = np.random.RandomState(seed)
    class_locations = {}
    foreground_mask = seg != 0
    foreground_coords = np.argwhere(foreground_mask)
    seg = seg[foreground_mask]
    del foreground_mask
    unique_labels = np.unique(seg.ravel())

    if isinstance(classes, int):
        classes = [classes]

    # We don't need more than 1e7 foreground samples. That's insanity. Cap here
    if len(foreground_coords) > 1e7:
        take_every = np.floor(len(foreground_coords) / 1e7)
        # keep computation time reasonable
        if verbose:
            print(f'Subsampling foreground pixels 1:{take_every} for computational reasons')
        foreground_coords = foreground_coords[::take_every]
        seg = seg[::take_every]

    for c in classes:
        if c not in unique_labels:
            class_locations[c] = []
            continue
        this_mask = seg == c
        locations_c = foreground_coords[this_mask]
        if not len(locations_c):
            class_locations[c] = []
            continue

        target_num_samples = min(num_samples, len(locations_c))

        target_num_samples = max(target_num_samples, int(np.ceil(len(locations_c) * min_percentage_covered)))

        locations_selected = locations_c[rndst.choice(len(locations_c), size=target_num_samples, replace=False)]
        class_locations[c] = locations_selected

        if verbose:
            print(c, target_num_samples)

        seg = seg[~this_mask]
        foreground_coords = foreground_coords[~this_mask]

    return class_locations


def get_normalizers() -> ty.Iterable[ty.Callable[[np.ndarray], np.ndarray]]:
    channel_0_norm = partial(zscore_norm, use_mask_for_norm=False, seg=None)
    channel_1_norm = partial(zscore_norm, use_mask_for_norm=False, seg=None)
    return [
        channel_0_norm,
        channel_1_norm
    ]


def get_resamplers() -> tuple[ty.Callable[[np.ndarray, tuple, tuple, tuple], np.ndarray],
                              ty.Callable[[np.ndarray, tuple, tuple, tuple], np.ndarray]]:
    shared_kwargs = {'order_z': 0, 'force_separate_z': None}
    resampling_data = partial(resample_data_or_seg_to_shape, is_seg=False, order=3, **shared_kwargs)
    resampling_seg = partial(resample_data_or_seg_to_shape, is_seg=True, order=1, **shared_kwargs)
    return resampling_data, resampling_seg


def preprocess_case(image_paths: ty.Iterable[str],
                    seg_path: str | None,
                    preprocessing_kwargs: dict):
    """
    Use this in inference.

    :param seg_path: can be `None` (inference)
    """

    data, properties = rw.read_images(image_paths)

    seg = seg_path
    if seg_path is not None:
        seg, _ = rw.read_seg(seg_path)

    return _preprocess_case(data, properties, seg=seg, **preprocessing_kwargs)


def preprocess_case_and_save(image_paths: tuple[str],
                             seg_path: str,
                             output_filename: str,
                             preprocessing_kwargs: dict,
                             clean_after: bool):
    """
    Use this to prepare for training.
    """
    data, seg, properties = preprocess_case(image_paths, seg_path, preprocessing_kwargs)
    Dataset.save_case(data, seg, properties, output_filename)
    if clean_after:
        for file in image_paths + (seg_path, ):
            remove(file)


def preprocess(
    data_iterable: dict[str, dict[str, ty.Iterable[str] | str | None]],
    dataset_fingerprint: str,
    output_folder: str,
    num_processes: int = 12,
    clean_after: bool = False,
    verbose: bool = True
):

    config = get_preprocessing_config_from_dataset_fingerprint(dataset_fingerprint)
    normalizers_per_channel = get_normalizers()
    resampler_data, resampler_seg = get_resamplers()

    kwargs = {
        'target_spacing': config.target_spacing,
        'transpose_forward': config.transpose_forward,
        'normalizers_per_channel': normalizers_per_channel,
        'resampling_function': resampler_data,
        'resampling_seg_function': resampler_seg,
        'verbose': verbose
    }

    maybe_mkdir(output_folder)

    results = []
    with multiprocessing.get_context('spawn').Pool(num_processes) as p:
        remaining = list(range(len(data_iterable)))
        workers = [j for j in p._pool]
        for k in data_iterable:
            args = [
                (data_iterable[k]['images'], data_iterable[k]['seg'], join(output_folder, 'case_' + k), kwargs, clean_after)
            ]
            results.append(p.starmap_async(preprocess_case_and_save, args))

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
            done = [i for i in remaining if results[i].ready()]
            # get done so that errors can be raised
            _ = [results[i].get() for i in done]
            for _ in done:
                results[_].get()  # allows triggering errors
            remaining = [i for i in remaining if i not in done]
            sleep(0.1)


def preprocess_entrypoint() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_data_folder', type=str,
                        help='Where all "identifiable" data is.')
    parser.add_argument('-d', '--dataset_fingerprint', type=str,
                        help='Path to dataset fingerprint. Required to know how preprocessing '
                             'should be approached. If not provided, fingerprint extraction '
                             'will be performed prior to preprocessing.')

    parser.add_argument('-o', '--output_folder', type=str,
                        help='Self-explanatory. Fuck you.')
    parser.add_argument('-np', '--num_processes', type=int,
                        help='Self-explanatory. F U.')
    parser.add_argument('--clean_after', action='store_true',
                        help='WARNING. Setting this will delete all data in "raw_data_folder" '
                             'after preprocessing has been run. This is here because I have '
                             'a humongous dataset. You probably don\'t... There\'s classes to this shit here.')

    parser.add_argument('--quiet', '-q', action='store_false',
                        dest='verbose')

    args = parser.parse_args()

    # handle some optional args
    args.output_folder = args.output_folder or join(dirname(args.raw_data_folder), 'preprocessed')
    args.num_processes = args.num_processes or get_default_num_processes() // 2

    if args.dataset_fingerprint is None:
        from src.fingerprint_extraction import run_and_save
        run_and_save(args.raw_data_folder,
                     args.raw_data_folder,
                     args.num_processes)
        args.dataset_fingerprint = join(args.raw_data_folder, 'dataset_fingerprint.json')

    data_iterable = generate_iterable_with_filenames(args.raw_data_folder, allow_no_seg=False)
    preprocess(data_iterable,
               args.dataset_fingerprint,
               args.output_folder,
               args.num_processes,
               args.clean_after,
               args.verbose)


if __name__ == "__main__":
    preprocess_entrypoint()
    # images = ('test-out/If_8948.nii.gz', 'test-out/Mb_8948.nii.gz')
    # seg = 'test-out/Mf_8948.nii.gz'
    # data_iterable = {'8948': {'images': images, 'seg': seg}}
    # data, seg, properties = preprocess(data_iterable, dataset_fingerprint='test-out/dataset_fingerprint.json')
    # for c in range(data.shape[0]):
    #     rw.write_image(data[c], output_fname=f'test_{c}.nii.gz', properties=properties)

    # rw.write_seg(seg[0], 'test_seg.nii.gz', properties)