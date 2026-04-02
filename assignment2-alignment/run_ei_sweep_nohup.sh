#!/usr/bin/env bash
set -euo pipefail

cd /scratch/pkk5421/Downloads/CSE587_NLP/assignment2-alignment

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/pkk5421/hf_cache
unset TRANSFORMERS_CACHE
export XDG_CACHE_HOME=/scratch/pkk5421/uv_cache
export TMPDIR=/scratch/pkk5421/tmp
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$HF_HOME" "$XDG_CACHE_HOME" "$TMPDIR" outputs/ei_math outputs/ei_math/logs

COMMON_ARGS=(
  --train-path data/MATH/train.jsonl
  --val-path data/MATH/validation.jsonl
  --model-name-or-path /scratch/pkk5421/models/Qwen2.5-Math-1.5B
  --hf-cache-dir /scratch/pkk5421/hf_cache
  --n-ei-steps 5
  --per-device-batch-size 1
  --gradient-accumulation-steps 32
  --rollout-batch-size 16
  --eval-batch-size 4
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.85
  --entropy-eval-examples 32
  --max-eval-tokens 256
  --bf16
  --gradient-checkpointing
)

run_exp () {
  local name="$1"
  shift
  echo "==== $(date) :: starting $name ===="
  .venv/bin/python scripts/run_expert_iteration_math.py "${COMMON_ARGS[@]}" "$@"
  echo "==== $(date) :: finished $name ===="
}

run_exp g4_e1_db512 \
  --output-dir outputs/ei_math/g4_e1_db512 \
  --rollout-count 4 \
  --sft-epochs 1 \
  --db-size 512

run_exp g8_e1_db512 \
  --output-dir outputs/ei_math/g8_e1_db512 \
  --rollout-count 8 \
  --sft-epochs 1 \
  --db-size 512

run_exp g4_e2_db512 \
  --output-dir outputs/ei_math/g4_e2_db512 \
  --rollout-count 4 \
  --sft-epochs 2 \
  --db-size 512

run_exp g4_e1_db1024 \
  --output-dir outputs/ei_math/g4_e1_db1024 \
  --rollout-count 4 \
  --sft-epochs 1 \
  --db-size 1024

run_exp g4_e1_db2048 \
  --output-dir outputs/ei_math/g4_e1_db2048 \
  --rollout-count 4 \
  --sft-epochs 1 \
  --db-size 2048

.venv/bin/python scripts/plot_ei_results.py \
  --experiment-dir outputs/ei_math/g4_e1_db512 \
  --experiment-dir outputs/ei_math/g8_e1_db512 \
  --experiment-dir outputs/ei_math/g4_e2_db512 \
  --experiment-dir outputs/ei_math/g4_e1_db1024 \
  --experiment-dir outputs/ei_math/g4_e1_db2048 \
  --output-dir outputs/ei_math/plots

.venv/bin/python - <<'PY'
import json, pathlib
exp_root = pathlib.Path("outputs/ei_math")
rows = []
for path in exp_root.glob("*/summary.json"):
    data = json.loads(path.read_text())
    rows.append((data["best_val_accuracy"], path.parent.name, data["best_val_step"], data["best_model_path"]))
rows.sort(reverse=True)
print("Best EI run:", rows[0])
for row in rows:
    print(row)
sft = json.loads(pathlib.Path("outputs/sft_qwen_1000_final/summary.json").read_text())
print("SFT baseline best_val_accuracy:", sft["best_val_accuracy"])
PY

