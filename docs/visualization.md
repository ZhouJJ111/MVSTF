# Real-data visualization

The included images and video show the 85.60 model's archived SemanticKITTI sequence 08 predictions. Their per-frame prediction and ground-truth hashes match `results/metrics85.json`. The manifests reference that report and its SHA256.

The new CPU renderer avoids a display-server/OpenGL dependency and writes three panels: ground truth, prediction, and errors. Red indicates moving points / true positives, blue false positives, and yellow false negatives. The displayed XY range is ±50 m; the frame counts and IoU use all original points, including those outside the displayed range. Overlapping points are projected onto pixels; no input points are subsampled. Ignored GT points do not count toward metrics. Per-frame input and image hashes are in `manifest.json`.

The 85.60 video uses frames 000000–000059 without selecting by score. The stills use 000000, 001000, 002000, 003000 and 004000. These are qualitative views of archived predictions, not live inference recordings or a substitute for full-sequence scoring.

To render your newly evaluated predictions, run from the repository root:

```bash
python tools/visualize_mos.py \
  --points "$MVSTF_DATA_ROOT/08/velodyne" \
  --ground-truth "$MVSTF_DATA_ROOT/08/labels" \
  --predictions outputs/semantickitti_85_60/val08/MOS_sequences/08 \
  --config datasets/semantic-kitti-mos.yaml \
  --metrics outputs/semantickitti_85_60/metrics.json \
  --frames 000000 001000 002000 003000 004000 \
  --output outputs/semantickitti_85_60/figures
```

For a video, first render a consecutive range of frame IDs, then use a separately installed FFmpeg:

```bash
ffmpeg -framerate 10 -start_number 0 -i path/to/rendered_frames/%06d.png \
  -frames:v 60 -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart demo.mp4
```

