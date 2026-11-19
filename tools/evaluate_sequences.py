#!/usr/bin/env python3
"""Score saved MOS predictions and print a compact terminal summary."""
import argparse
import json
from pathlib import Path
import numpy as np
from evaluate_mos_strict import evaluate, summarize, sha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root', type=Path, required=True, help='Directory containing <sequence>/labels')
    ap.add_argument('--predictions-root', type=Path, required=True, help='Directory containing <sequence>/<prediction-subdir>')
    ap.add_argument('--sequences', nargs='+', required=True)
    ap.add_argument('--prediction-subdir', default='predictions', help='Use . for predictions directly under each sequence')
    ap.add_argument('--prediction-encoding', choices=['raw-semantic', 'learning'], required=True)
    ap.add_argument('--config', type=Path, default=Path('datasets/semantic-kitti-mos.yaml'))
    ap.add_argument('--name', required=True, help='Name of the saved prediction archive being scored')
    ap.add_argument('--output', type=Path, required=True, help='Fresh directory for full per-sequence reports')
    args = ap.parse_args()
    if len(set(args.sequences)) != len(args.sequences):
        ap.error('Duplicate sequence IDs')
    if any(len(s) != 2 or not s.isdigit() for s in args.sequences):
        ap.error('Sequence IDs must have two digits')
    args.output.mkdir(parents=True, exist_ok=False)
    matrix = np.zeros((2, 2), dtype=np.int64)
    reports = []
    frames = points = ignored = 0
    for seq in args.sequences:
        print('Scoring sequence ' + seq, flush=True)
        result = evaluate(args.data_root / seq / 'labels', args.predictions_root / seq / args.prediction_subdir,
                          args.config, args.prediction_encoding)
        path = args.output / (seq + '.json')
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        reports.append({'sequence': seq, 'file': path.name, 'sha256': sha(path.read_bytes())})
        matrix += np.asarray(result['results']['overall']['confusion_rows_gt_columns_prediction'], dtype=np.int64)
        frames += result['frames']
        points += result['total_points']
        ignored += result['ignored_gt_points']
    overall = summarize(matrix)
    combined = dict(name=args.name, sequences=args.sequences, frames=frames, total_points=points,
                    ignored_gt_points=ignored, overall=overall, reports=reports,
                    aggregation='sum confusion matrices over every listed sequence',
                    runner_sha256=sha(Path(__file__).read_bytes()),
                    scope='CPU scoring of saved predictions; no model inference executed')
    path = args.output / 'summary.json'
    path.write_text(json.dumps(combined, indent=2, allow_nan=False) + '\n')
    moving = overall['classes']['moving']
    def pct(value):
        return 'undefined' if value is None else f'{100 * value:.4f}'
    print('\n' + '=' * 76)
    print('MVSTF | CPU evaluation of saved predictions')
    print('Archive: ' + args.name)
    print('Sequences: ' + ', '.join(args.sequences))
    print(f'Frames: {frames:,} | Points: {points:,} | Ignored GT: {ignored:,}')
    print(f"Moving TP: {moving['tp']:,} | FP: {moving['fp']:,} | FN: {moving['fn']:,}")
    print(f"{'IoU (%)':>12} {'Precision (%)':>16} {'Recall (%)':>14} {'Mean IoU (%)':>16}")
    print(f"{pct(moving['iou']):>12} {pct(moving['precision']):>16} {pct(moving['recall']):>14} {pct(overall['mean_iou']):>16}")
    print('Report: ' + str(path))
    print('SHA256: ' + sha(path.read_bytes()))
    print('No new GPU inference is performed by this command.')
    print('=' * 76, flush=True)


if __name__ == '__main__':
    main()
