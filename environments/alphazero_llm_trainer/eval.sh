#!/bin/bash
# evaluation script - correctness, tokens, perplexity only (no student model)

# set api key and url
export PRIME_API_KEY="${PRIME_API_KEY:-pit_8cddd0aacc3c728659344298b7d53ffe9e959832de8f6ff8bfc04986cd17fc22}"
export PRIME_INFERENCE_URL="${PRIME_INFERENCE_URL:-https://api.pinference.ai/api/v1}"
export TERMINAL_CHECKER_MODEL="${TERMINAL_CHECKER_MODEL:-google/gemini-2.0-flash-lite-001}"

# default values
NUM_EXAMPLES=${1:-20}
MODEL=${2:-"google/gemini-3-pro-preview"}
ROLLOUTS=${3:-3}

echo "=" * 80
echo "alphazero llm trainer - correctness evaluation"
echo "=" * 80
echo "examples: $NUM_EXAMPLES"
echo "model: $MODEL"
echo "rollouts: $ROLLOUTS"
echo "=" * 80

# create outputs directory
mkdir -p ./outputs

# run evaluation using vf-eval
uv run vf-eval alphazero-llm-trainer \
  --model "$MODEL" \
  --num-examples "$NUM_EXAMPLES" \
  --rollouts-per-example "$ROLLOUTS" \
  --env-args "{\"use_student_model\": false, \"use_combined\": false}" \
  --api-base-url "$PRIME_INFERENCE_URL" \
  --api-key-var "PRIME_API_KEY" \
  --save-dataset

echo "evaluation complete"