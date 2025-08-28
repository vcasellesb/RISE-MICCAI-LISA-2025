import shlex
from glob import glob


from synthsr import synthsr_main


def run_synthsr_on_lowfield_scan(lowfield_scan: str, output_file: str, nthreads: int = 4) -> None:
    """
    Usage:

    mri_synthsr --i <input> --o <output> --threads <n_threads> [--v1] [--lowfield] [--ct]
    From: https://www.freesurfer.net/fswiki/SynthSR
    """
    lowfield_scan = shlex.quote(lowfield_scan)
    output_file = shlex.quote(output_file)

    command = 'mri_synthsr --i %s --o %s --threads %i --lowfield' % (
        lowfield_scan,
        output_file,
        nthreads
    )
    command = shlex.split(command)
    return synthsr_main(command[1:])

DATASET_PATH = 'test_upscaling'
def run_on_all_dataset():
    low_field_ims = glob(DATASET_PATH + '/unnormalizedbm3d.nii.gz')
    for low_field_im in low_field_ims:
        output_file = low_field_im.replace('.nii.gz', '_SR.nii.gz')
        run_synthsr_on_lowfield_scan(low_field_im, output_file)

if __name__ == "__main__":
    run_on_all_dataset()