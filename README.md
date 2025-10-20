# AlphaZero LLM Trainer

**For Prime Intellect** - GitHub backup repository for development tracking.

See the `environments/alphazero_llm_trainer/` folder for implementation details.

---

## Overview

This project implements an AlphaZero-inspired training framework for Large Language Models (LLMs) using Monte Carlo Tree Search (MCTS) combined with modern RL techniques. The system trains a student model to solve mathematical reasoning tasks (GSM8K dataset) by exploring reasoning paths guided by teacher ensembles and reward signals.

**Key Components:**
- **MCTS-based exploration** for structured reasoning path generation
- **Teacher ensemble** using multiple LLMs (DeepSeek, Qwen, Llama) via vLLM
- **Student model** fine-tuned with GRPO (Group Relative Policy Optimization)
- **Dual reward system**: Hard Reward Estimator (HRE) + Perplexity Reward Estimator (PRE)
- **Terminal state detection** for efficient tree pruning

---

## Project Structure

```
alphazero-llm-trainer/
├── pyproject.toml                     # Root project configuration
├── README.md                          # This file
└── environments/
    └── alphazero_llm_trainer/         # Main implementation
        ├── config/                    # Configuration files
        │   ├── models.yaml            # Student/teacher model definitions
        │   ├── teacher_models.yaml    # Teacher ensemble config
        │   └── training.yaml          # MCTS, reward, training hyperparameters
        │
        ├── core/                      # Core algorithmic components
        │   ├── environment.py         # Verifiers integration (SingleTurnEnv)
        │   ├── mcts.py                # MCTS search algorithm with UCT
        │   └── tree.py                # Tree node structure
        │
        ├── models/                    # Model implementations
        │   ├── student_model.py       # Unsloth-based student (Qwen2.5-3B)
        │   ├── teacher_ensemble_vllm.py  # vLLM teacher ensemble
        │   └── terminal_checker_vllm.py  # Response completion detector
        │
        ├── rewards/                   # Reward computation
        │   ├── combined.py            # Weighted HRE + PRE combination
        │   ├── hre.py                 # Hard Reward (correctness)
        │   ├── pre.py                 # Perplexity Reward (teacher agreement)
        │   └── rubric_functions.py    # Evaluation rubrics
        │
        ├── prompts/                   # Prompt templates
        │   ├── terminal_check.py      # Terminal state detection prompt
        │   └── token_gen.py           # Token generation guidance
        │
        ├── utils/                     # Utilities
        │   ├── answer_extraction.py   # Parse numeric answers
        │   ├── dataset_loader.py      # GSM8K loading
        │   └── similarity.py          # Response similarity metrics
        │
        ├── unsloth_compiled_cache/    # Cached Unsloth trainers
        │
        ├── train_vllm.py              # Main training script
        ├── test.py                    # Evaluation script
        └── requirements.txt           # Dependencies
```

---

## Theoretical Foundation

### 1. **AlphaZero for Language**
Based on [Silver et al., 2017](https://arxiv.org/abs/1712.01815), adapted from game tree search to sequential reasoning generation. Instead of board states, nodes represent partial text responses; actions are token sequences (10 "right" + 10 "wrong" alternatives per expansion).

### 2. **Monte Carlo Tree Search (MCTS)**
- **Selection**: UCT (Upper Confidence Bound for Trees) balances exploitation (high rewards) and exploration (low visit counts)
- **Expansion**: Generate token alternatives using teacher models and student policy
- **Simulation**: Evaluate partial/complete responses using dual reward system
- **Backpropagation**: Update visit counts and Q-values up the tree

### 3. **Group Relative Policy Optimization (GRPO)**
Inspired by [PPO](https://arxiv.org/abs/1707.06347) and [REINFORCE](https://link.springer.com/article/10.1007/BF00992696) with group normalization:
- Normalize advantages relative to trajectory groups (reduces variance)
- KL penalty term keeps student close to reference policy
- Advantages clipped to prevent outlier gradients

### 4. **Teacher Ensemble Distillation**
Multiple teacher models provide diverse reasoning paths. Perplexity-based reward measures consensus (lower perplexity = higher teacher agreement = better reasoning quality).

### 5. **Reward Shaping**
- **HRE (Hard)**: Binary correctness (+1 for exact answer match, 0 otherwise)
- **PRE (Soft)**: Negative log-perplexity from teacher ensemble (encourages fluent, teacher-like reasoning)
- **Combined**: Weighted sum with length penalty and direction bonus (favors "right" branches)

---

## Key Features

1. **vLLM Integration**: High-throughput inference for teacher ensemble (DeepSeek-R1-Distill, Qwen2.5-72B, Llama-3.3-70B)
2. **Unsloth Optimization**: Memory-efficient LoRA fine-tuning with 4-bit quantization
3. **Binary Tree Exploration**: Each node spawns 2 children ("right"=10 tokens, "wrong"=10 tokens) with binary codes for tracking
4. **Terminal State Caching**: LLM-based terminal checker prevents unnecessary expansions
5. **Checkpointing**: Save/resume training with model snapshots every N examples

---

## Usage

**Setup:**
```bash
cd environments/alphazero_llm_trainer
pip install -r requirements.txt
export OPENROUTER_API_KEY="your_key_here"
```

**Train:**
```bash
python train_vllm.py \
  --num-examples 350 \
  --checkpoint-dir ./checkpoints \
  --save-every 50 \
  --use-grpo
```

**Test:**
```bash
python test.py
```

---

## Configuration

Edit `config/training.yaml` to adjust:
- MCTS iterations, exploration constant, tree depth
- Reward weights (HRE vs PRE)
- GRPO hyperparameters (beta, advantage normalization)
- Student learning rate, batch size, gradient accumulation

---

## Related Papers

- **AlphaZero**: [Silver et al., 2017 - Mastering Chess and Shogi by Self-Play with a General RL Algorithm](https://arxiv.org/abs/1712.01815)
- **PPO**: [Schulman et al., 2017 - Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **MCTS Survey**: [Browne et al., 2012 - A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)
- **Teacher Ensemble**: [Hinton et al., 2015 - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- **GSM8K Dataset**: [Cobbe et al., 2021 - Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)

---

## License

See individual component licenses. This backup is for Prime Intellect development tracking.
