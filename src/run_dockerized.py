from glob import glob

from .utils import join
from .inference_entrypoint import predict_from_files

## Specifications
# input files will be mounted into a directory called /input
# output files must be saved at /output
def main(input_folder: str, output_folder: str):
    input_files = glob(join(input_folder, '*.nii.gz'))
    output_files = pass