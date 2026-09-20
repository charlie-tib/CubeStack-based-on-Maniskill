#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
entrypoint="src/ppo_stackcube.py"

common_args=(
  --experiment b4_norm
  --env-id StackCube-v1
  --learning-rate 3e-4
  --gamma 0.8
  --gae-lambda 0.9
  --clip-coef 0.2
  --target-kl 0.2
  --total-timesteps 10000000
  --num-envs 512
  --num-steps 50
  --num-minibatches 32
  --update-epochs 4
  --num-eval-envs 16
  --num-eval-steps 50
  --eval-freq 10
  --no-partial-reset
)

for train_seed in 1 2 3; do
  run_name="StackCube-v1__B4-Norm-NoEarlyReset__seed${train_seed}"
  checkpoint="runs/${run_name}/final_ckpt.pt"
  log_dir="runs/${run_name}/eval_logs"

  if [[ -e "$checkpoint" ]]; then
    echo "refusing to overwrite existing checkpoint: $checkpoint" >&2
    exit 1
  fi

  PYTHONPATH=. "$python_bin" "$entrypoint" \
    --exp-name "$run_name" \
    --seed "$train_seed" \
    "${common_args[@]}"

  test -s "$checkpoint"
  mkdir -p "$log_dir"
  for eval_seed in 101 202 303; do
    PYTHONPATH=. "$python_bin" "$entrypoint" \
      --exp-name "${run_name}__eval_${eval_seed}" \
      --evaluate \
      --checkpoint "$checkpoint" \
      --seed "$train_seed" \
      --eval-seed "$eval_seed" \
      --eval-episodes 100 \
      --no-capture-video \
      "${common_args[@]}" 2>&1 | tee "$log_dir/seed_${eval_seed}.log"
  done
done
