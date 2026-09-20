#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python}"
mode="${1:-smoke}"
label="B4-Norm-NoEarlyReset-TCAPS"

# Two-factor combination: the B4-Norm baseline plus the independently tested
# NoEarlyReset termination treatment and temporal CAPS actor regularization.
common_args=(
  --experiment b4_norm
  --env-id StackCube-v1
  --seed 1
  --learning-rate 3e-4
  --gamma 0.8
  --gae-lambda 0.9
  --clip-coef 0.2
  --target-kl 0.2
  --eval-freq 10
  --no-partial-reset
  --caps-temporal-coef 0.1
)

smoke() {
  local run_name checkpoint
  run_name="smoke__${label}__seed1__$(date +%s)"
  checkpoint="runs/${run_name}/final_ckpt.pt"

  "$python_bin" src/ppo_stackcube.py \
    --exp-name "$run_name" \
    --total-timesteps 256 \
    --num-envs 32 \
    --num-steps 8 \
    --num-minibatches 4 \
    --update-epochs 1 \
    --num-eval-envs 4 \
    --num-eval-steps 4 \
    --no-capture-video \
    "${common_args[@]}"

  test -s "$checkpoint"
  "$python_bin" src/ppo_stackcube.py \
    --exp-name "${run_name}__roundtrip_eval" \
    --evaluate \
    --checkpoint "$checkpoint" \
    --eval-seed 101 \
    --eval-episodes 4 \
    --total-timesteps 256 \
    --num-envs 32 \
    --num-steps 8 \
    --num-minibatches 4 \
    --update-epochs 1 \
    --num-eval-envs 4 \
    --num-eval-steps 50 \
    --no-capture-video \
    "${common_args[@]}"

  echo "smoke passed: $label"
}

full() {
  local run_name checkpoint log_dir eval_seed
  run_name="StackCube-v1__${label}__seed1"
  checkpoint="runs/${run_name}/final_ckpt.pt"
  log_dir="runs/${run_name}/eval_logs"

  if [[ -e "$checkpoint" ]]; then
    echo "refusing to overwrite existing checkpoint: $checkpoint" >&2
    return 1
  fi

  "$python_bin" src/ppo_stackcube.py \
    --exp-name "$run_name" \
    --total-timesteps 10000000 \
    --num-envs 512 \
    --num-steps 50 \
    --num-minibatches 32 \
    --update-epochs 4 \
    --num-eval-envs 16 \
    --num-eval-steps 50 \
    "${common_args[@]}"

  test -s "$checkpoint"
  mkdir -p "$log_dir"
  for eval_seed in 101 202 303; do
    "$python_bin" src/ppo_stackcube.py \
      --exp-name "${run_name}__eval_${eval_seed}" \
      --evaluate \
      --checkpoint "$checkpoint" \
      --eval-seed "$eval_seed" \
      --eval-episodes 100 \
      --total-timesteps 10000000 \
      --num-envs 512 \
      --num-steps 50 \
      --num-minibatches 32 \
      --update-epochs 4 \
      --num-eval-envs 16 \
      --num-eval-steps 50 \
      --no-capture-video \
      "${common_args[@]}" 2>&1 | tee "$log_dir/seed_${eval_seed}.log"
  done
}

case "$mode" in
  smoke)
    smoke
    ;;
  full)
    full
    ;;
  *)
    echo "usage: $0 [smoke|full]" >&2
    exit 2
    ;;
esac
