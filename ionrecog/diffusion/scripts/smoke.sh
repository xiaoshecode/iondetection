#!/usr/bin/env bash
set -euo pipefail
# Run from Net Recog repository root with the environment activated.
out="${1:-runs/diffusion_smoke}"
python -m ionrecog.diffusion synthetic --output "$out/data" --train-groups 3 --val-groups 1 --test-groups 1 --size 32
python -m ionrecog.diffusion validate --manifest "$out/data/manifest.jsonl"
python -m ionrecog.diffusion train --manifest "$out/data/manifest.jsonl" --config ionrecog/diffusion/configs/smoke.yaml --output "$out/train" --device cpu
python -m ionrecog.diffusion evaluate --manifest "$out/data/manifest.jsonl" --checkpoint "$out/train/best.pt" --output "$out/evaluate" --device cpu --samples 2 --save-images
