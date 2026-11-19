#!/usr/bin/env python3
"""Check trusted local checkpoint keys and shapes on CPU; does not run inference."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(str(args.output))
    cfg = importlib.import_module(args.config.replace('.py', '').replace('/', '.'))
    params = cfg.get_config()[2]
    module, name = params.prefix.rsplit('.', 1)
    model = getattr(importlib.import_module('models.' + module), name)(params)
    # torch 1.10 does not support weights_only. Only use trusted author checkpoints.
    state = torch.load(str(args.checkpoint), map_location='cpu')
    result = model.load_state_dict(state, strict=True)
    report = {'status': 'strict_cpu_load_pass', 'config': args.config, 'model': params.prefix,
              'checkpoint': args.checkpoint.name,
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'torch': torch.__version__, 'state_dict_entries': len(state),
              'missing_keys': list(result.missing_keys), 'unexpected_keys': list(result.unexpected_keys),
              'forward_executed': False, 'cuda_kernels_executed': False,
              'scope': 'CPU parameter loading only; not inference or numerical reproduction'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
