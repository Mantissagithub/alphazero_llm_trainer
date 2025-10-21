# alphazero-llm-trainer

### Overview
- **Environment ID**: `alphazero-llm-trainer`
- **Short description**: AlphaZero-style MCTS-guided training for LLMs on mathematical reasoning tasks
- **Tags**: math, reasoning, reinforcement-learning, mcts, grpo, eval, gsm8k, llm


### Datasets
- **Primary dataset(s)**: GSM8K (Grade School Math 8K) - mathematical word problems with natural language solutions
- **Source links**: `openai/gsm8k` on HuggingFace Datasets
- **Split sizes**: Configurable (default: 50 train, 20 eval)

### Task
- **Type**: single-turn
- **Parser**: Custom answer extraction with regex parsing for numerical answers (####-prefixed format)
- **Rubric overview**: Combined reward system with Hard Reward Estimator (HRE, 40%) for correctness and Perplexity Reward Estimator (PRE, 60%) for trajectory quality

### Quickstart

**Run an evaluation** with default settings:

```bash
uv run vf-eval alphazero-llm-trainer
```

**Run the training script** (trains the student model using MCTS + GRPO/Legacy mode):

```bash
vf-eval alphazero-llm-trainer -s
```

Or with custom training arguments:

```bash
vf-eval alphazero-llm-trainer -s -- \
  --num-examples 100 \
  --checkpoint-dir ./checkpoints \
  --save-every 25 \
  --use-grpo
```

Configure model and evaluation:

```bash
uv run vf-eval alphazero-llm-trainer \
  --tier free \
  --use-student-model
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `tier` | str | `"free"` | Model tier for teacher ensemble (free/production) |
| `use_student_model` | bool | `False` | Enable student model for perplexity-based rewards |

### Training Script Arguments

The training script (`train_vllm.py`) can be run with `vf-eval alphazero-llm-trainer -s` and supports the following arguments:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `--num-examples` | int | 50 (from config) | Number of training examples to use |
| `--checkpoint-dir` | str | from config | Directory to save model checkpoints |
| `--save-every` | int | from config | Save checkpoint every N examples |
| `--log-interval` | int | from config | Log progress every N examples |
| `--use-grpo` | flag | from config | Enable GRPO training mode |
| `--use-legacy` | flag | False | Use legacy reward-weighted supervised learning |

**Example:**
```bash
vf-eval alphazero-llm-trainer -s -- --num-examples 200 --use-grpo --save-every 50
```

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted combination of correctness and perplexity) |
| `correctness_reward` | Binary reward (1.0 for correct answer, 0.0 otherwise) from HRE |
| `perplexity_reward` | Normalized perplexity-based reward from PRE (only if student model enabled) |

### How It Works

This environment implements AlphaZero-style reinforcement learning for training language models on mathematical reasoning:

**MCTS Search System:**
- Performs Monte Carlo Tree Search with configurable iterations (default: 60)
- Expands tree nodes using teacher ensemble or student model predictions
- Uses UCB1 selection with exploration constant (√2)
- Maximum tree depth of 12 levels with binary expansion (right/wrong tokens)

**Reward System:**
- **Hard Reward Estimator (HRE)**: Terminal state checker that validates numerical answers against ground truth using regex extraction
- **Perplexity Reward Estimator (PRE)**: Measures trajectory quality using student model perplexity, normalized against baseline
- **Combined Rewards**: Weighted combination (40% HRE, 60% PRE) with length penalties and direction bonuses

**Training Components:**
- **Student Model**: Llama 3.2 3B with 4-bit quantization and LoRA fine-tuning (r=32)
- **Teacher Ensemble**: Multiple free-tier models (DeepSeek R1, GPT-OSS, Llama 3.3, Qwen3) for diverse reasoning paths
- **Terminal Checker**: Nvidia Nemotron Nano 9B for fast terminal state detection
- **GRPO Training**: Group Relative Policy Optimization with KL penalty and advantage clipping

**Configuration:**
- All hyperparameters in `config/training.yaml`, `config/models.yaml`, `config/teacher_models.yaml`
- Supports both GRPO and legacy training modes
- Configurable MCTS parameters (iterations, depth, exploration)
- Adjustable reward weights and training settings

