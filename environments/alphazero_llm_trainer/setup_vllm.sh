#!/bin/bash
# setup_vllm.sh - Complete vLLM setup for Prime Intellect H100

set -e

echo "=========================================="
echo "AlphaZero LLM - vLLM Setup"
echo "=========================================="

# System packages
echo "Installing system dependencies..."
apt-get update -qq
apt-get install -y git htop nvtop vim curl wget

# Install UV if not present
if ! command -v uv &> /dev/null; then
    echo "Installing UV package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Create Python environment
echo "Creating Python 3.11 environment..."
cd /workspace/alphazero-llm-trainer
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
echo "Installing Python packages..."
uv pip install -r requirements.txt

# Install flash-attention separately
echo "Building flash-attention (takes 5-10 min)..."
uv pip install flash-attn --no-build-isolation

# Download models
echo "=========================================="
echo "Downloading Models (15-20 minutes)"
echo "=========================================="

python3 << 'EOF'
from huggingface_hub import snapshot_download
import os

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

models = [
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "Qwen/Qwen2.5-0.5B-Instruct"
]

for i, model in enumerate(models, 1):
    print(f"\n[{i}/{len(models)}] {model}")
    snapshot_download(repo_id=model, resume_download=True)
    print("✓")

print("\n" + "=" * 60)
print("✓ All models cached")
print("=" * 60)
EOF

echo "=========================================="
echo "✓ Setup Complete!"
echo "Run: python train_vllm.py"
echo "=========================================="
