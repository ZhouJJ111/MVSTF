#!/usr/bin/env python3
"""Strict, CPU-only MOS audit evaluator. This is not a hidden-test evaluator."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml


def sha(data):
    return hashlib.sha256(data).hexdigest()


def label_files(directory):
    files = {p.stem: p for p in directory.glob('*.label')}
    if not files:
        raise ValueError(f'No .label files in {directory}')
    if any(len(stem) != 6 or not stem.isdigit() for stem in files):
        raise ValueError(f'Expected six-digit frame identifiers in {directory}')
    return files


def read_array(path, dtype):
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise OSError(f'Cannot read binary array {path}: {exc}') from exc
    if not data or len(data) % np.dtype(dtype).itemsize:
        raise ValueError(f'Empty or malformed binary array: {path}')
    return np.frombuffer(data, dtype=dtype), sha(data)


def remap_raw(values, mapping, path):
    raw = values & 0xFFFF
    found = np.unique(raw)
    unknown = [int(v) for v in found if int(v) not in mapping]
    if unknown:
        raise ValueError(f'Unknown raw label IDs in {path}: {unknown}')
    lut = np.zeros(max(mapping) + 1, dtype=np.uint8)
    for key, value in mapping.items():
        lut[int(key)] = int(value)
    return lut[raw]


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def summarize(matrix):
    ious = []
    classes = {}
    for i, name in enumerate(['static', 'moving']):
        tp = int(matrix[i, i])
        fn = int(matrix[i, :].sum()) - tp
        fp = int(matrix[:, i].sum()) - tp
        iou = ratio(tp, tp + fp + fn)
        ious.append(iou)
        classes[name] = dict(tp=tp, fp=fp, fn=fn, support=tp + fn,
                             iou=iou, precision=ratio(tp, tp + fp), recall=ratio(tp, tp + fn))
    return dict(confusion_rows_gt_columns_prediction=matrix.tolist(), classes=classes,
                mean_iou=float(np.mean(ious)) if all(v is not None for v in ious) else None,
                valid_points=int(matrix.sum()), metric_scale='fraction_0_to_1',
                undefined_metric_policy='null; never silently replace undefined values with zero')


def evaluate(gt_dir, pred_dir, config, encoding, points_dir=None, distance_metric=None):
    if points_dir is not None and distance_metric not in ['xy', 'xyz']:
        raise ValueError('A distance metric (xy or xyz) is required with points')
    cfg_bytes = config.read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    mapping = {int(k): int(v) for k, v in cfg['learning_map'].items()}
    if set(mapping.values()) != {0, 1, 2}:
        raise ValueError('Expected MOS mapping: 0=ignored, 1=static, 2=moving')
    inverse = {int(k): int(v) for k, v in cfg['learning_map_inv'].items()}
    if inverse.get(1) not in mapping or inverse.get(2) not in mapping or mapping[inverse[1]] != 1 or mapping[inverse[2]] != 2:
        raise ValueError('Invalid MOS inverse mapping')
    if not cfg.get('learning_ignore', {}).get(0, False):
        raise ValueError('Expected GT class 0 to be ignored')
    if any(cfg.get('learning_ignore', {}).get(k, False) for k in [1, 2]):
        raise ValueError('Static and moving classes must both be evaluated')
    gt_files, pred_files = label_files(gt_dir), label_files(pred_dir)
    if set(gt_files) != set(pred_files):
        raise ValueError(f'Frame mismatch: missing={sorted(set(gt_files)-set(pred_files))[:10]}, extra={sorted(set(pred_files)-set(gt_files))[:10]}')
    if encoding not in ['raw-semantic', 'learning']:
        raise ValueError('Prediction encoding must be explicitly specified')
    if points_dir is not None:
        point_files = {p.stem: p for p in points_dir.glob('*.bin')}
        if set(point_files) != set(gt_files):
            raise ValueError('Point/label frame identifiers differ')
    names = ['overall'] + (['close_lt20', 'medium_ge20_lt50', 'far_ge50'] if points_dir else [])
    matrices = {name: np.zeros((2, 2), dtype=np.int64) for name in names}
    manifest = []
    total_points = ignored_points = 0
    for frame_number, frame in enumerate(sorted(gt_files), 1):
        gt_raw, gt_sha = read_array(gt_files[frame], '<u4')
        pred_raw, pred_sha = read_array(pred_files[frame], '<u4')
        if gt_raw.size != pred_raw.size:
            raise ValueError(f'Point-count mismatch in frame {frame}: GT={gt_raw.size}, prediction={pred_raw.size}')
        gt = remap_raw(gt_raw, mapping, gt_files[frame])
        if encoding == 'raw-semantic':
            pred = remap_raw(pred_raw, mapping, pred_files[frame])
        else:
            if np.any(pred_raw > 2):
                raise ValueError(f'Learning labels must be 0,1,2: {pred_files[frame]}')
            pred = pred_raw.astype(np.uint8)
        valid = gt != 0
        # Do not discard points where a model outputs ignored/unlabeled on valid GT.
        if np.any(valid & (pred == 0)):
            raise ValueError(f'Ignored prediction on valid GT in frame {frame}; audit label encoding/export before scoring')
        total_points += int(gt.size)
        ignored_points += int((~valid).sum())
        masks = {'overall': valid}
        record = dict(frame=frame, points=int(gt.size), gt_sha256=gt_sha, pred_sha256=pred_sha)
        if points_dir:
            points, points_sha = read_array(point_files[frame], '<f4')
            if points.size != gt.size * 4:
                raise ValueError(f'Expected four float32 values per point: {point_files[frame]}')
            points = points.reshape(-1, 4)
            dims = 2 if distance_metric == 'xy' else 3
            coordinates = points[:, :dims].astype(np.float64)
            if not np.all(np.isfinite(coordinates)):
                raise ValueError(f'Nonfinite point coordinates: {point_files[frame]}')
            distances = np.linalg.norm(coordinates, axis=1)
            masks.update(close_lt20=valid & (distances < 20),
                         medium_ge20_lt50=valid & (distances >= 20) & (distances < 50),
                         far_ge50=valid & (distances >= 50))
            record['points_sha256'] = points_sha
        for name, mask in masks.items():
            indices = (gt[mask].astype(np.int64) - 1) * 2 + (pred[mask].astype(np.int64) - 1)
            matrices[name] += np.bincount(indices, minlength=4).reshape(2, 2)
        manifest.append(record)
        if frame_number % 250 == 0:
            print(f'Evaluated {frame_number}/{len(gt_files)} frames; latest={frame}', flush=True)
    if not matrices['overall'].sum():
        raise ValueError('No valid ground-truth points; cannot report MOS performance')
    if points_dir and not np.array_equal(sum(matrices[k] for k in names[1:]), matrices['overall']):
        raise AssertionError('Distance partitions do not sum to overall confusion matrix')
    return dict(status='evaluated', frames=len(manifest), total_points=total_points,
                ignored_gt_points=ignored_points, prediction_encoding=encoding,
                distance_metric=distance_metric, config=str(config), config_sha256=sha(cfg_bytes),
                evaluator_sha256=sha(Path(__file__).read_bytes()),
                results={k: summarize(v) for k, v in matrices.items()}, frame_manifest=manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth', type=Path, required=True, help='One sequence labels directory')
    parser.add_argument('--predictions', type=Path, required=True, help='Matching .label directory')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--prediction-encoding', required=True, choices=['raw-semantic','learning'])
    parser.add_argument('--points', type=Path)
    parser.add_argument('--distance-metric', choices=['xy','xyz'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.ground_truth, args.predictions, args.config, args.prediction_encoding, args.points, args.distance_metric)
    # No output is written unless every frame passes verification.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing result: {args.output}')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'frames':result['frames'], 'overall':result['results']['overall'], 'output':str(args.output)}))


if __name__ == '__main__':
    main()
