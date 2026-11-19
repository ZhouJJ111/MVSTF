# Evaluation results

The release candidate preserves historical checkpoint identities and evaluation records. A CPU score of archived predictions establishes those predictions' score; it does not by itself certify training provenance or prove that this release candidate produced every archived prediction.

The historical sequence-08 audit contains 4,071 frames and 499,079,562 points (12,895,907 ignored ground-truth points). Moving-class counts are TP=1,474,250, FP=58,124 and FN=189,871, giving 85.6004807678% moving IoU. The checkpoint identity is recorded in `checkpoint_best/weights_manifest.json`. Three previously unreadable predictions were regenerated with the historical checkpoint; three neighboring readable predictions matched byte-for-byte. The full historical training protocol remains unverified.

`results/metrics85.json` and `.log` are the actual CPU re-scoring output for the 85.60 archive. The corresponding 84.51 archive result is `results/metrics84.json`, with its separate log. `results/summary.png` renders these real result reports for convenient viewing; it is not a desktop screenshot or a GPU recording. The JSON files include per-frame hashes and counts. No new GPU inference or training was run for these CPU scoring reports.

Both SemanticKITTI checkpoints passed strict CPU loading with their own model configurations, without missing or unexpected keys. The 84.51 checkpoint has 649 state-dict entries and uses MAFL without the later additional BatchNorm; 85.60 has 669 entries and uses MAFL_BN. The original common-config approach fails strict loading for 84.51. The new separate variant restores the corresponding archived code without converting weights or ignoring keys. This is a loading check, not a forward-pass validation or certification of the full historical training recipe.

For a new recording on a working CUDA machine:

1. Start a terminal screen recording and show the code revision, `nvidia-smi`, Python/PyTorch versions and checkpoint filename.
2. Run the README inference command into a fresh output directory. Show the config, dataset split, FP16 and TTA settings.
3. Run the strict CPU evaluator against the newly exported predictions. Keep the full log and result JSON, including per-frame hashes, with the recording.
4. Show the final frame count and metrics without cutting away errors. A short qualitative clip can supplement these numeric results.

For a raw terminal transcript, `script -q -e evaluation_session.log` starts recording. Run the commands inside that shell and type `exit` when finished. Use a screen recorder separately if a video is desired. Run this procedure separately to record a new session.

The bundled reports use paths relative to the repository root. The `config` field names the included label mapping, and the log `output` field names the corresponding result JSON. These publication path adjustments preserve all scores, counts and per-frame input hashes.

## Paper Tables II and VII: terminal evaluation

The [paper (arXiv v1)](https://arxiv.org/html/2611.20872v1) reports SemanticKITTI metrics in Table II and KITTI-Road moving IoU in Table VII. The following commands score saved MVSTF predictions. They print IoU, precision, recall and mean IoU in percent after all requested frames pass strict evaluation. They execute CPU scoring, without running model inference. Comparison methods require their own prediction archives.

For Table II, set `MVSTF_PREDICTIONS_85_ROOT` and `MVSTF_PREDICTIONS_84_ROOT` to the respective directories containing `08/predictions/*.label`, then run from the repository root:

```bash
python tools/evaluate_sequences.py \
  --data-root "$MVSTF_DATA_ROOT" \
  --predictions-root "$MVSTF_PREDICTIONS_85_ROOT" \
  --sequences 08 --prediction-encoding raw-semantic \
  --name 'MVSTF 85.60 | Table II | SemanticKITTI validation' \
  --output "outputs/table2_85_60_$(date +%Y%m%d_%H%M%S)"

python tools/evaluate_sequences.py \
  --data-root "$MVSTF_DATA_ROOT" \
  --predictions-root "$MVSTF_PREDICTIONS_84_ROOT" \
  --sequences 08 --prediction-encoding raw-semantic \
  --name 'MVSTF 84.51 | Table II | SemanticKITTI validation' \
  --output "outputs/table2_84_51_$(date +%Y%m%d_%H%M%S)"
```

For Table VII, first generate Road predictions with the [KITTI-Road inference command](../README.md#6-kitti-road-inference-and-evaluation). To score an existing archive, `MVSTF_ROAD_DATA_ROOT` must contain the KITTI-Road validation labels under `<sequence>/labels`, and `MVSTF_ROAD_PREDICTIONS_ROOT` must contain predictions from the Road model under `<sequence>/predictions`. Confirm the mapping uses raw IDs 9 (static) and 251 (moving) before running:

```bash
python tools/evaluate_sequences.py \
  --data-root "$MVSTF_ROAD_DATA_ROOT" \
  --predictions-root "$MVSTF_ROAD_PREDICTIONS_ROOT" \
  --sequences 35 36 37 38 39 41 --prediction-encoding raw-semantic \
  --name 'MVSTF | Table VII | KITTI-Road validation' \
  --output "outputs/table7_road_$(date +%Y%m%d_%H%M%S)"
```

The Road command pools all six confusion matrices before computing IoU. It does not average sequence IoUs. Each run saves full per-sequence JSON reports and a combined `summary.json`. For the README inference output layout (`<sequence>/*.label`), add `--prediction-subdir .` and use `<run>/kitti_road/MOS_sequences` as the predictions root. The Road command requires predictions from the separate Road checkpoint; the bundled SemanticKITTI reports do not reproduce Table VII.

Capture the terminal after the final summary appears, including the archive name, sequences, frame count and metric columns. Full reports remain available alongside the screenshot.
