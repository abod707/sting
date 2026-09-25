#!/usr/bin/env bash
# sting3 — end-to-end pipeline: data -> fine-tune -> build -> evaluate.
#
# Compute reality on a 1-OCPU ARM box: ~90 s/step at batch 8, seq 384. The full
# 4,810-example corpus is ~541 steps ≈ 13 h for one epoch. LoRA trains at the full
# 20 layers (there is no per-rung fine-tune: `needle finetune --layers` is advertised
# in help but not implemented, cactus-compute/needle#139), so rung choice happens at
# `build` time, after training.
set -euo pipefail

VENV=/home/ubuntu/.venvs/needle
PY="$VENV/bin/python"
NEEDLE="$VENV/bin/needle"
HERE=$(cd "$(dirname "$0")" && pwd)
BASE="$HERE/checkpoints/checkpoints/needle3.safetensors"
STAGE=${1:-all}

cd "$HERE"
export NEEDLE_TELEMETRY=0

generate_data() {
  echo "== data =="
  "$PY" gen_data.py --out data/data.jsonl
  # Fit the sequence budget by trimming distractor tools, never by truncating the
  # answer tokens (--max-len truncation silently drops target tokens from the loss).
  "$PY" fit_dataset.py data/data.jsonl data/data384.jsonl --budget 384
  "$PY" measure_lens.py data/data384.jsonl | head -12
}

finetune() {
  echo "== fine-tune (LoRA r16, 20L, 1 epoch) =="
  # Run outside the Hermes worker cgroup: that scope caps memory at ~4 GB and the
  # trainer peaks at ~5.6 GB (JAX holds the model in float32 plus activations).
  systemd-run --user --unit=sting3-train -p MemoryMax=9G -p CPUWeight=25 \
    -p WorkingDirectory="$HERE" \
    "$NEEDLE" finetune data/data384.jsonl \
      --checkpoint "$BASE" \
      --epochs 1 --batch-size 8 --lr 1e-4 --lora-rank 16 --lora-alpha 32 \
      --max-len 384 --val-split 0.1 --seed 0 \
      --out checkpoints/sting3_lora.safetensors --checkpoint-dir checkpoints
  echo "   follow with: journalctl --user -u sting3-train -f"
}

build() {
  echo "== build rungs (W4A8) =="
  for L in 4 8 20; do
    "$NEEDLE" build "$BASE" --lora checkpoints/sting3_lora.safetensors \
      --layers "$L" --out "engines/sting3-${L}L.cact"
  done
  ls -la engines/
}

evaluate() {
  echo "== evaluate base vs tuned =="
  ENGINE=/home/ubuntu/.hermes/cache/scratch/n3/needle
  [ -x "$ENGINE" ] || ENGINE="$VENV/bin/needle"
  "$PY" eval_model.py --model "$HERE/../../.cache/cactus-needle/v3/3.0.1/needle3.cact" \
    --tools agent_tools.json --label base --out results_base.json || true
  for L in 4 8 20; do
    [ -f "engines/sting3-${L}L.cact" ] || continue
    "$PY" eval_model.py --model "engines/sting3-${L}L.cact" \
      --tools agent_tools.json --label "sting3-${L}L" --out "results_${L}L.json" || true
  done
}

case "$STAGE" in
  data)      generate_data ;;
  train)     finetune ;;
  build)     build ;;
  eval)      evaluate ;;
  all)       generate_data; finetune ;;
  *)         echo "usage: $0 [data|train|build|eval|all]"; exit 1 ;;
esac
