#!/bin/bash
# Quick evaluation script for alphazero-llm-trainer (Prime Env Hub Style, fixed flags)

# Set API key, URL, and terminal checker (required for your env)
export PRIME_API_KEY="${PRIME_API_KEY:-pit_8cddd0aacc3c728659344298b7d53ffe9e959832de8f6ff8bfc04986cd17fc22}"
export PRIME_INFERENCE_URL="${PRIME_INFERENCE_URL:-https://api.pinference.ai/api/v1}"
export TERMINAL_CHECKER_MODEL="${TERMINAL_CHECKER_MODEL:-google/gemini-2.0-flash-lite-001}"  # Ensures terminal checks work

# Default values
NUM_EXAMPLES=${1:-10}
MODEL=${2:-"moonshotai/kimi-k2-thinking"}
TEMPERATURE=${3:-0.1}
HF_DATASET_NAME=${4:-""}  # Optional: e.g., "your_username/alphazero-gsm8k-eval"

echo "=" * 80
echo "AlphaZero LLM Trainer - Evaluation (Prime Env Hub Style)"
echo "=" * 80
echo "Examples: $NUM_EXAMPLES"
echo "Model: $MODEL"
echo "Terminal Checker: $TERMINAL_CHECKER_MODEL"
echo "Temperature: $TEMPERATURE"
echo "HF Dataset (if set): $HF_DATASET_NAME"
echo "=" * 80

# Create outputs/ folder (like wordle-go)
mkdir -p ./outputs

# Timestamp for final filename
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SAVE_PATH="./outputs/eval_${TIMESTAMP}.json"

# Default verifiers save location
DEFAULT_SAVE="./eval_dataset.json"

# Run evaluation (valid flags only)
if uv run vf-eval alphazero-llm-trainer \
  --model "$MODEL" \
  --api-base-url "$PRIME_INFERENCE_URL" \
  --api-key-var "PRIME_API_KEY" \
  --env-args "{\"use_student_model\": false, \"use_combined\": true, \"num_train_examples\": 500, \"num_eval_examples\": $NUM_EXAMPLES}" \
  --num-examples 100 \
  --rollouts-per-example 5 \
  --temperature 0.1 \
  --max-concurrent 32 \
  --verbose \
  --save-dataset; then  # Valid flag: saves to ./eval_dataset.json

  # Move/rename if successful (hub-style post-processing)
  if [ -f "$DEFAULT_SAVE" ]; then
    mv "$DEFAULT_SAVE" "$SAVE_PATH"
    echo "Results saved to: $SAVE_PATH"
  else
    echo "Warning: Default save not found; check logs for issues."
  fi

  # Optional HF upload
  if [ -n "$HF_DATASET_NAME" ] && [ -f "$SAVE_PATH" ]; then
    echo "Uploading to HF Hub: $HF_DATASET_NAME"
    uv run python -c "
import datasets
ds = datasets.load_dataset('json', data_files='$SAVE_PATH', split='train')
ds.push_to_hub('$HF_DATASET_NAME')
    "
    echo "Uploaded: https://huggingface.co/datasets/$HF_DATASET_NAME"
  fi

  echo "View with: vf-tui $SAVE_PATH"  # Inspect rollouts locally
else
  echo "Eval failed; no results saved. Check args/logs above."
  exit 1
fi
