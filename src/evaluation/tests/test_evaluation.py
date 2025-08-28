import re
import subprocess

from src.utils import load_json


def seg_stats_dice(mask1: str, mask2: str):
    return subprocess.check_output(['seg_stats', mask1, '-d', mask2])

def parse_seg_stats(seg_stats_output: bytes) -> str:
    return seg_stats_output.decode('utf-8')

def load_metrics_per_case(json_path: str) -> list[dict]:
    return [m for m in load_json(json_path)['metric_per_case'] if m['baseline_file'] is None]

def _regex(pattern):
    regex = re.compile(pattern)
    def inner(string: str):
        return regex.search(string).groups()[-1]
    return inner

def process_case(case_dict: dict):
    mask1 = case_dict['gt_file']
    mask2 = case_dict['prediction_file']
    mydice = case_dict['metrics']['Dice']

    seg_stats_dc = parse_seg_stats(seg_stats_dice(mask1, mask2))
    return seg_stats_dc, mydice

def main(json_path):
    metrics_per_case = load_metrics_per_case(json_path)
    regex = _regex(r'Label\[1\]\s+=\s+(\d+(?:\.\d+)?)')
    for case_dict in metrics_per_case:
        seg_stats_dc, mydice = process_case(case_dict)
        assert float(regex(seg_stats_dc)) == float(mydice)

if __name__ == "__main__":
    myjson='timelessegv2_trained_models_06082025_210859/final_validation_results.json'
    main(myjson)