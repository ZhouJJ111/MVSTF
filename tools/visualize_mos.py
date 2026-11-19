#!/usr/bin/env python3
"""Render real point clouds, GT and archived MOS predictions on a CPU (no OpenGL)."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--points', type=Path, required=True)
    ap.add_argument('--ground-truth', type=Path, required=True)
    ap.add_argument('--predictions', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--metrics', type=Path, required=True, help='Full prior strict evaluation JSON, with per-frame hashes')
    ap.add_argument('--frames', nargs='+', required=True, help='Explicit six-digit frame IDs')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--title', default='Archived MVSTF predictions')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory: ' + str(args.output))
    cfg_bytes = args.config.read_bytes()
    cfg = yaml.safe_load(cfg_bytes)
    metrics_bytes = args.metrics.read_bytes()
    metrics = json.loads(metrics_bytes)
    if metrics['config_sha256'] != sha(cfg_bytes) or metrics['prediction_encoding'] != 'raw-semantic':
        raise ValueError('Result config or prediction encoding differs')
    records = {x['frame']: x for x in metrics['frame_manifest']}
    mapping = {int(k): int(v) for k, v in cfg['learning_map'].items()}
    lut = np.zeros(max(mapping) + 1, dtype=np.uint8)
    for key, value in mapping.items():
        lut[key] = value
    args.output.mkdir(parents=True)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
    font = ImageFont.truetype(font_path, 17) if Path(font_path).exists() else ImageFont.load_default()
    manifest = {'title': args.title, 'kind': 'CPU BEV rendering of archived predictions; not live inference', 'metrics_sha256': sha(metrics_bytes), 'config_sha256': sha(cfg_bytes), 'view': 'x/y in [-50,50] m; forward x is up; counts use all points; no point subsampling', 'frames': []}
    for frame in args.frames:
        if len(frame) != 6 or not frame.isdigit() or frame not in records:
            raise ValueError('Invalid or unchecked frame: ' + frame)
        point_bytes = (args.points / (frame + '.bin')).read_bytes()
        gt_bytes = (args.ground_truth / (frame + '.label')).read_bytes()
        pred_bytes = (args.predictions / (frame + '.label')).read_bytes()
        expected = records[frame]
        for key, data in [('gt_sha256', gt_bytes), ('pred_sha256', pred_bytes), ('points_sha256', point_bytes)]:
            if key in expected and sha(data) != expected[key]:
                raise ValueError('Hash mismatch: ' + frame + ' ' + key)
        points = np.frombuffer(point_bytes, dtype='<f4').reshape(-1, 4)
        gt_raw = np.frombuffer(gt_bytes, dtype='<u4') & 0xFFFF
        pred_raw = np.frombuffer(pred_bytes, dtype='<u4') & 0xFFFF
        if len(points) != len(gt_raw) or len(points) != len(pred_raw) or len(points) != expected['points']:
            raise ValueError('Point-count mismatch: ' + frame)
        if not np.isfinite(points[:, :3]).all():
            raise ValueError('Nonfinite coordinates: ' + frame)
        for values in [gt_raw, pred_raw]:
            if not set(np.unique(values)).issubset(mapping):
                raise ValueError('Unknown raw class')
        gt, pred = lut[gt_raw], lut[pred_raw]
        valid = gt != 0
        if np.any(valid & (pred == 0)):
            raise ValueError('Ignored prediction on valid GT')
        tp = int(np.sum(valid & (gt == 2) & (pred == 2)))
        fp = int(np.sum(valid & (gt == 1) & (pred == 2)))
        fn = int(np.sum(valid & (gt == 2) & (pred == 1)))
        iou = tp / (tp + fp + fn) if tp + fp + fn else None
        shown = (np.abs(points[:, 0]) <= 50) & (np.abs(points[:, 1]) <= 50)
        px = np.clip(((50 - points[:, 1]) * 5.99).astype(int), 0, 599)
        py = np.clip(((50 - points[:, 0]) * 5.99).astype(int), 0, 599)
        image = Image.new('RGB', (1840, 820), '#121a24')
        draw = ImageDraw.Draw(image)
        draw.text((20, 12), args.title + ' | sequence 08 | frame ' + frame, font=font, fill='white')
        score = 'undefined' if iou is None else f'{100 * iou:.4f}%'
        draw.text((20, 42), f'All-point counts: N={len(points):,}  TP={tp:,} FP={fp:,} FN={fn:,}  frame moving IoU={score}', font=font, fill='white')
        for panel, title in enumerate(['Ground truth', 'Prediction', 'Errors on valid GT']):
            canvas = np.full((600, 600, 3), (18, 26, 36), dtype=np.uint8)
            if panel < 2:
                label = gt if panel == 0 else pred
                groups = [(shown & (label == 0), (45, 55, 65)), (shown & (label == 1), (150, 160, 170)), (shown & (label == 2), (255, 70, 70))]
            else:
                groups = [(shown & valid, (120, 135, 145)), (shown & valid & (gt == 2) & (pred == 2), (255, 70, 70)), (shown & valid & (gt == 1) & (pred == 2), (65, 170, 255)), (shown & valid & (gt == 2) & (pred == 1), (255, 220, 40))]
            for mask, color in groups:
                canvas[py[mask], px[mask]] = color
            image.paste(Image.fromarray(canvas), (10 + panel * 610, 110))
            draw.text((20 + panel * 610, 82), title, font=font, fill='white')
        draw.text((20, 725), 'Red: moving / TP    Blue: false positive    Yellow: false negative    Gray: static', font=font, fill='white')
        draw.text((20, 751), 'BEV: x,y within +/-50 m; forward x points up. Counts use ALL original points.', font=font, fill='white')
        draw.text((20, 777), 'Archived predictions checked against per-frame hashes; CPU rendering, no new GPU inference.', font=font, fill='white')
        output = args.output / (frame + '.png')
        image.save(output)
        manifest['frames'].append({'frame': frame, 'points': len(points), 'displayed_points': int(shown.sum()), 'tp': tp, 'fp': fp, 'fn': fn, 'moving_iou': iou, 'points_sha256': sha(point_bytes), 'gt_sha256': sha(gt_bytes), 'pred_sha256': sha(pred_bytes), 'image_sha256': sha(output.read_bytes())})
        print(frame + ' rendered', flush=True)
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
