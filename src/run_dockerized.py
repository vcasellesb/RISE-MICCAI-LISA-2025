import argparse
import re
from glob import glob

from .utils import join, maybe_mkdir
from .inference_entrypoint import predict_from_files

## Specifications
# input files will be mounted into a directory called /input
# output files must be saved at /output

REGEX_IDENTIFIER = r'LISA_.*?_(\d{4})_CISO.nii.gz'
def construct_output_filenames(input_files: list[str], output_folder: str) -> list[str]:
    pat = re.compile(REGEX_IDENTIFIER, re.I)
    identifiers = map(lambda x: x.groups()[-1], filter(None, map(pat.search, input_files)))
    return list(map(lambda x: join(output_folder, 'LISA_HF_%s_seg_prediction.nii.gz' % x), identifiers))

def main(input_folder: str, output_folder: str,
         models_path: str | None,
         num_processes_preprocesing: int,
         num_processes_export: int,
         nthreads_synthsr: int,
         npp_hdbet: int,
         npe_hdbet: int,
         tile_step_size_hdbet: float,
         use_gaussian_hdbet: bool,
         use_tta_hdbet: bool):

    input_files = glob(join(input_folder, '*.nii.gz'))
    output_files = construct_output_filenames(input_files, output_folder)
    maybe_mkdir(output_folder)
    predict_from_files(input_files, output_files,
                       models_path=models_path,
                       num_processes_prep=num_processes_preprocesing,
                       num_processes_export=num_processes_export,
                       nthreads_synthsr=nthreads_synthsr,
                       num_processes_prep_hdbet=npp_hdbet,
                       num_processes_export_hdbet=npe_hdbet,
                       tile_step_size_hdbet=tile_step_size_hdbet,
                       use_gaussian_hdbet=use_gaussian_hdbet,
                       use_tta_hdbet=use_tta_hdbet)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', '-i', type=str, default='/input')
    parser.add_argument('--output_folder', '-o', type=str, default='/output')
    parser.add_argument('--model_path', '-mp', type=str)

    parser.add_argument('--num_processes_preprocesing', '-npp', type=int,
                        default=3)
    parser.add_argument('--num_processes_export', '-npe', type=int,
                        default=3)

    # SYNTHSR STUFF (SUPER RESOLUTION)
    parser.add_argument('--nthreads_synthsr', type=int, default=2)

    # HD BET STUFF (BRAIN SEGMENTATION)
    parser.add_argument('-npp_hdbet', type=int, default=3)
    parser.add_argument('-npe_hdbet', type=int, default=3)
    parser.add_argument('--tile_step_size_hdbet', type=float, default=0.5)
    parser.add_argument('--no_gaussian_hdbet', action='store_false')
    parser.add_argument('--use_tta_hdbet', action='store_true')

    args = parser.parse_args()

    main(args.input_folder, args.output_folder, args.model_path, args.num_processes_preprocesing, args.num_processes_export,
         nthreads_synthsr=args.nthreads_synthsr,
         npp_hdbet=args.npp_hdbet,
         npe_hdbet=args.npe_hdbet,
         tile_step_size_hdbet=args.tile_step_size_hdbet,
         use_gaussian_hdbet=args.no_gaussian_hdbet,
         use_tta_hdbet=args.use_tta_hdbet)