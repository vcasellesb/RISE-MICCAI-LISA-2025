import re
import numpy as np

from src.utils import (
    listdir,
    join,
    exists,
    isfile,
    write_pickle,
    load_pickle
)

from .unpack_dataset import unpack_dataset


LISA_REGEX = r'(?:LISA|case)_(\d{4})_?.*(?:\.nii(?:\.gz)?|\.np(?:z|y))'

def _get_identifiers(folder: str) -> set[str]:
    data_regex = re.compile(LISA_REGEX)
    return set(
        map(
            lambda x: x.groups()[-1],
            filter(None, map(
                data_regex.search,
                listdir(folder)
                )
            )
        )
    )

def get_identifiers(folder: str, sort: bool = True) -> list[str]:
    identifiers = list(_get_identifiers(folder))
    if sort:
        identifiers.sort()
    return identifiers


def _get_image_fnames(folder: str, identifier: str, file_ending: str) -> tuple[str, str]:
    image_fnames = (
        'LISA_%s_LF.nii.gz',
        'LISA_%s_SR.nii.gz'
    )
    image_fnames = tuple(map(lambda x: join(folder, x % identifier), image_fnames))
    return image_fnames



def _generate_iterable_with_fnames(folder: str, identifier: str, allow_no_seg: bool, file_ending: str) -> dict[str, str | tuple[str, str]]:
    """
    :param allow_no_seg: enforces non-available ground truths.
    """

    image_fnames = _get_image_fnames(folder, identifier, file_ending)

    seg_fname = join(folder, 'LISA_%s_GT.nii.gz' % identifier)
    if not exists(seg_fname):
        if not allow_no_seg:
            raise FileNotFoundError('seg "%s" not found' % seg_fname)

        seg_fname = None

    assert all(exists(f) for f in image_fnames)

    return {'images': image_fnames, 'seg': seg_fname}


def generate_iterable_with_filenames(
    folder: str,
    allow_no_seg: bool = True,
    file_ending: str = '.nii.gz'
) -> dict[str, dict]:

    return {ii: _generate_iterable_with_fnames(folder, ii, allow_no_seg, file_ending) for ii in get_identifiers(folder)}


class Dataset:
    suffix: str = '.npz'
    suffix_props: str = '.pkl'
    suffix_uncompressed: str = '.npy'
    def __init__(
        self,
        folder: str
    ):
        super().__init__()
        self.folder = folder
        self.identifiers: list[str] = get_identifiers(folder)

    def load_case(self, identifier: str) -> tuple[np.ndarray, np.ndarray, dict]:
        """Both data and seg are returned as 4D."""
        identifier = 'case_' + identifier

        data_uncompressed_file = join(self.folder, identifier + self.suffix_uncompressed)
        if isfile(data_uncompressed_file):
            data = np.load(data_uncompressed_file, mmap_mode='r')
        else:
            data = np.load(join(self.folder, identifier + self.suffix))['data']

        seg_uncompressed_file = join(self.folder, identifier + '_seg' + self.suffix_uncompressed)
        if isfile(seg_uncompressed_file):
            seg = np.load(seg_uncompressed_file, mmap_mode='r')
        else:
            seg = np.load(join(self.folder, identifier + self.suffix))['seg']

        properties = load_pickle(join(self.folder, identifier + self.suffix_props))
        return data, seg, properties


    @staticmethod
    def save_case(
        data: np.ndarray,
        seg: np.ndarray,
        properties: dict,
        output_filename_truncated: str
    ):
        np.savez_compressed(output_filename_truncated + Dataset.suffix, data=data, seg=seg)
        write_pickle(properties, output_filename_truncated + Dataset.suffix_props)


    def unpack_dataset(self, num_processes: int, remove_npz: bool):
        npz_files = list(map(lambda idd: join(self.folder, f'case_{idd}{self.suffix}'), self.identifiers))
        unpack_dataset(npz_files, num_processes, unpack_segmentation=True, verify_npy=True, remove_npz=remove_npz)


if __name__ == "__main__":
    assert (
        _get_identifiers('datasets/2025Task2/Low Field Images') ==
        _get_identifiers('datasets/2025Task2/Subtask 2a - Hippocampus Segmentations') ==
        _get_identifiers('datasets/2025Task2/Subtask 2b - Basal Ganglia Segmentations')
        and len(_get_identifiers('datasets/2025Task2/Low Field Images')) == 79
    )

    assert (len(generate_iterable_with_filenames('training_data/raw', False)) == 45)