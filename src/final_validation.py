from .config import get_preprocessing_config
from .data_stuff import VALIDATION_PATH, LABELS

from .utils import join, dirname, get_default_device, get_default_num_processes

from .dataloading.dataset import Dataset
from .inference.predictor import Predictor
from .trainer.final_validation import final_validation
from .evaluation.metrics import compute_metrics_on_folder


def final_validation_entrypoint():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder', type=str, metavar='model_folder')

    parser.add_argument('-np', '--num_processes', type=int,
                        default=get_default_num_processes())

    parser.add_argument('--device', type=str, default=get_default_device())

    parser.add_argument('--checkpoint', type=str, default='final',
                        choices=['latest', 'best', 'final'])
    parser.add_argument('--only_compute_metrics', action='store_true')

    args = parser.parse_args()

    return args


def handle_checkpoint(checkpoint: str, output_folder: str):
    return join(output_folder, 'checkpoint_%s.pth' % checkpoint)

def construct_dict_with_preds(validation_data_path: str, models_folder: str) -> dict:
    val_dataset = Dataset(validation_data_path)
    predictions_folder = join(models_folder, 'final_validation_results')
    return {i: join(predictions_folder, 'case_%s.nii.gz' % i) for i in val_dataset.identifiers}

def main():
    args = final_validation_entrypoint()
    preprocessing_config = get_preprocessing_config()
    path_to_weights = handle_checkpoint(args.checkpoint, args.output_folder)
    predictor = Predictor.from_checkpoint_path(path_to_weights, preprocessing_config, device=args.device)

    if args.only_compute_metrics:
        dict_with_preds = construct_dict_with_preds(VALIDATION_PATH, args.output_folder)
    else:
        dict_with_preds = final_validation(predictor, VALIDATION_PATH,
                                           args.num_processes,
                                           join(args.output_folder, 'final_validation_results'),
                                           preprocessing_config)

    compute_metrics_on_folder(join(dirname(VALIDATION_PATH), 'raw'),
                              dict_with_preds,
                              join(args.output_folder, 'final_validation_results.json'),
                              LABELS[1:], # filter bg
                              args.num_processes)

if __name__ == "__main__":
    main()