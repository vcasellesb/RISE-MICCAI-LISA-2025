import multiprocessing
import numpy as np
from collections import defaultdict

from src.utils import save_json
from src.io import rw
from src.dataloading.dataset import generate_iterable_with_filenames

from .json_export import recursive_fix_for_json_export


def compute_tp_fp_fn_tn(
    gt: np.ndarray,
    pred: np.ndarray
) -> tuple[int, int, int, int]:

    # true positives:
    # voxels that are lesion and
    # have been predicted as such
    tp = np.sum(gt & pred)

    # false positives:
    # voxels that are not lesion (positive)
    # BUT have been predicted as such
    fp = np.sum(~gt & pred)

    # false negatives:
    # voxels that are actually lesion
    # but have not been predicted correctly
    fn = np.sum(gt & ~pred)

    # true negatives:
    # voxels that are "healthy"
    # and are predicted as such
    tn = np.sum(~gt & ~pred)

    return tp, fp, fn, tn


def compute_metrics(gt_file: str, prediction_file: str, labels: list[int]) -> dict:
    # load images
    gt, gt_properties = rw.read_seg(gt_file)
    pred, pred_properties = rw.read_seg(prediction_file)
    assert gt.shape == pred.shape, gt_file

    results = {}
    results['gt_file'] = gt_file
    results['prediction_file'] = prediction_file

    results['metrics'] = defaultdict(dict)

    for label in labels:
        this_gt = gt == label
        this_pred = pred == label
        tp, fp, fn, tn = compute_tp_fp_fn_tn(this_gt, this_pred)

        denom = tp + fp + fn
        if denom == 0:
            results['metrics'][label]['Dice'] = results['metrics']['IoU'] = np.nan
        else:
            results['metrics'][label]['Dice'] = 2 * tp / (2 * denom)
            results['metrics'][label]['IoU'] = tp / denom

        results['metrics'][label]['FP'] = fp
        results['metrics'][label]['TP'] = tp
        results['metrics'][label]['FN'] = fn
        results['metrics'][label]['TN'] = tn
        results['metrics'][label]['n_pred'] = fp + tp
        results['metrics'][label]['n_ref'] = fn + tp

    return results


def compute_metrics_on_folder(gt_folder: str,
                              dict_with_preds: dict[str, str],
                              output_file: str | None,
                              labels: list[int],
                              num_processes: int):
    """
    output_file must end with .json; can be None
    """
    iterable_gts = generate_iterable_with_filenames(gt_folder, allow_no_seg=False, file_ending='.nii.gz')
    identifiers = set(iterable_gts.keys())
    assert identifiers == set(dict_with_preds.keys())

    with multiprocessing.get_context("spawn").Pool(num_processes) as pool:
        results = pool.starmap(
            compute_metrics,
            [
                (iterable_gts[identifier]['seg'], dict_with_preds[identifier], labels)
                for identifier in identifiers
            ]
        )

    metric_list = list(results[0]['metrics'][labels[0]].keys())
    # mean metric per class
    means = {}
    for label in labels:
        means[label] = {}
        for m in metric_list:
            means[label][m] = np.nanmean([i['metrics'][label][m] for i in results])

    # foreground mean
    foreground_mean = {}
    for m in metric_list:
        values = []
        for k in means.keys():
            if k == 0 or k == '0':
                continue
            values.append(means[k][m])
        foreground_mean[m] = np.mean(values)

    result = {'metric_per_case': results, 'mean': means, 'foreground_mean': foreground_mean}
    recursive_fix_for_json_export(result)

    if output_file is not None:
        save_json(result, output_file)

    return result


if __name__ == "__main__":
    folder = 'validation_data/raw'
    dict_with_preds = {k: v['seg'] for k, v in generate_iterable_with_fnames(folder).items()}
    compute_metrics_on_folder(gt_folder=folder,
                              dict_with_preds=dict_with_preds,
                              output_file='test.json', num_processes=16)