# AlphaZero LLM Trainer

**For Prime Intellect** - GitHub backup repository for development tracking.

See the `environments/alphazero_llm_trainer/` folder for implementation details.

Prime Intellect Environment link: [pradheep/alphazero-llm-trainer](https://app.primeintellect.ai/dashboard/environments/pradheep/alphazero-llm-trainer)

---

## Overview

This project implements an AlphaZero-inspired training framework for Large Language Models (LLMs) using Monte Carlo Tree Search (MCTS) combined with modern RL techniques. The system is fully integrated with the **Prime Intellect Verifiers framework** and trains a student model to solve mathematical reasoning tasks (GSM8K dataset) by exploring reasoning paths guided by teacher ensembles and reward signals.

**Key Components:**
- **Verifiers Integration**: Environment extends `MultiTurnEnv` with rubric-based evaluation
- **MCTS-based exploration** for structured reasoning path generation
- **Teacher ensemble** using multiple LLMs (DeepSeek, Qwen, Llama) via vLLM
- **Student model** (Llama 3.2-3B) fine-tuned with GRPO (Group Relative Policy Optimization)
- **Dual reward system**: Hard Reward Estimator (HRE) + Perplexity Reward Estimator (PRE)
- **Terminal state detection** with vLLM-based checker for efficient tree pruning
- **Adaptive tree depth** and compute scaling via environment variables

---

## Project Structure

```
alphazero-llm-trainer/
├── pyproject.toml                     # Root project configuration
├── README.md                          # This file
└── environments/
    └── alphazero_llm_trainer/         # Main implementation (Verifiers environment)
        ├── alphazero_llm_trainer.py   # Verifiers environment registration
        │
        ├── config/                    # Configuration files
        │   ├── __init__.py            # Config loaders
        │   ├── models.yaml            # Student/teacher model definitions
        │   ├── teacher_models.yaml    # Teacher ensemble config (free/production tiers)
        │   └── training.yaml          # MCTS, reward, GRPO hyperparameters
        │
        ├── core/                      # Core algorithmic components
        │   ├── environment.py         # AlphaZeroLLMEnvironment (extends MultiTurnEnv)
        │   ├── mcts.py                # MCTS search with UCT and trajectory collection
        │   └── tree.py                # TreeNode with UCT/PUCT scoring
        │
        ├── models/                    # Model implementations
        │   ├── student_model.py       # Unsloth-based student (Llama-3.2-3B with LoRA)
        │   ├── teacher_ensemble_vllm.py  # vLLM teacher ensemble with adaptive sampling
        │   └── terminal_checker_vllm.py  # vLLM-based response completion detector
        │
        ├── rewards/                   # Reward computation
        │   ├── combined.py            # Weighted HRE + PRE with length/direction bonuses
        │   ├── hre.py                 # Hard Reward (correctness-based)
        │   ├── pre.py                 # Perplexity Reward (teacher consensus)
        │   └── rubric_functions.py    # Verifiers-compatible rubric functions
        │
        ├── prompts/                   # Prompt templates
        │   ├── terminal_check.py      # Terminal state detection prompt
        │   └── token_gen.py           # Token generation guidance (right/wrong)
        │
        ├── utils/                     # Utilities
        │   ├── answer_extraction.py   # Parse numeric answers
        │   ├── dataset_loader.py      # GSM8K loading with Verifiers schema
        │   └── similarity.py          # Response similarity metrics
        │
        ├── unsloth_compiled_cache/    # Cached Unsloth trainers (auto-generated)
        │
        ├── train_vllm.py              # Main training script with GRPO/Legacy modes
        ├── test.py                    # Evaluation script (config, models, environment)
        ├── test_gsm8k.py              # GSM8K specific tests
        ├── test_config_only.py        # Standalone config tests
        ├── requirements.txt           # Dependencies
        ├── pyproject.toml             # Package metadata + Verifiers config
        └── setup_vllm.sh              # vLLM setup helper
```

---

## Theoretical Foundation

### 1. **AlphaZero for Language**
Based on [Silver et al., 2017](https://arxiv.org/abs/1712.01815), adapted from game tree search to sequential reasoning generation. Instead of board states, nodes represent partial text responses; actions are token sequences (10 "right" + 10 "wrong" alternatives per expansion).

### 2. **Monte Carlo Tree Search (MCTS)**
- **Selection**: UCT (Upper Confidence Bound for Trees) balances exploitation (high rewards) and exploration (low visit counts)
- **Expansion**: Generate token alternatives using teacher models and student policy with similarity-based filtering
- **Simulation**: Evaluate partial/complete responses using dual reward system with accumulated perplexity
- **Backpropagation**: Update visit counts and Q-values up the tree, with teacher model weight updates
- **Trajectory Collection**: Gather (state, next_state, reward) tuples for policy learning

### 3. **Group Relative Policy Optimization (GRPO)**
Inspired by [PPO](https://arxiv.org/abs/1707.06347) and [REINFORCE](https://link.springer.com/article/10.1007/BF00992696) with group normalization:
- Normalize advantages relative to trajectory groups (reduces variance)
- KL penalty term keeps student close to reference policy (prevents distribution collapse)
- Reference model periodically updated from student weights
- Advantages clipped to prevent outlier gradients
- Legacy mode: Reward-weighted supervised learning for ablation studies

### 4. **Verifiers Framework Integration**
Built on Prime Intellect's [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) framework:
- **MultiTurnEnv**: Environment base class with async rollout support
- **Rubric**: Composable reward function system with weighted combinations
- **Dataset Schema**: Standardized prompt/answer/info structure
- **Completion Logic**: Custom `is_completed()` with depth limits and token budgets
- **Environment Registration**: `load_environment()` factory pattern

### 5. **Teacher Ensemble Distillation**
Multiple teacher models provide diverse reasoning paths:
- **Tier-based Selection**: Free tier (DeepSeek R1-Distill, Nemotron) vs production (Qwen2.5-72B, Llama-3.3-70B)
- **Adaptive Sampling**: Teacher outputs used for "right" vs "wrong" branch generation
- **Perplexity Consensus**: Lower perplexity = higher teacher agreement = better reasoning quality
- **Dynamic Weighting**: Teacher model performance tracked and used for future sampling

### 6. **Reward Shaping**
- **HRE (Hard)**: Binary correctness (+1 for exact answer match, 0 otherwise)
- **PRE (Soft)**: Negative log-perplexity from student model (encourages fluent, teacher-like reasoning)
- **Accumulated PRE**: Discounted sum of perplexity rewards along path (γ=0.95)
- **Length Penalty**: -0.01 × depth (prevents overly long reasoning chains)
- **Direction Bonus**: +0.1 per "right" branch, -0.05 per "wrong" branch
- **Combined**: Weighted sum with configurable HRE/PRE balance (default: 0.4/0.6)

---

---

## Architecture & Implementation Details

### Environment Integration (Verifiers)

The `AlphaZeroLLMEnvironment` extends `verifiers.MultiTurnEnv` to provide:

**Dataset Schema:**
```python
{
    'prompt': str,      # Question text (GSM8K)
    'answer': str,      # Reference answer
    'info': {
        'reference': str  # Additional metadata
    }
}
```

**Rubric Composition:**
```python
rubric = vf.Rubric(
    funcs=[correctness_reward, perplexity_reward],
    weights=[hre_weight, pre_weight]
)
```

**Completion Logic:**
- Depth limit: Controlled by `MAX_TREE_DEPTH` environment variable
- Terminal state: Detected by vLLM-based checker
- Token budget: Optional `MAX_TOKENS` limit
- MCTS exhaustion: No more promising branches to explore

### MCTS Implementation

**Tree Node Structure:**
```python
class TreeNode:
    state: str                    # Accumulated response text
    binary_code: str              # Path tracking (1=right, 0=wrong)
    visits: int                   # Visit count for UCT
    value: float                  # Accumulated reward
    accumulated_pre: float        # Discounted perplexity sum
    step_pre: float               # Current step perplexity
    student_similarity: float     # Similarity to student output
```

**UCT Selection:**
```python
uct_score = (value / visits) + c * sqrt(log(parent.visits) / visits)
```

**Expansion Strategy:**
1. Generate student token continuation (ε-greedy)
2. Query teacher ensemble for 5 "right" continuations
3. Query teacher ensemble for 5 "wrong" continuations
4. Calculate similarity between student and teacher outputs
5. Select best child based on similarity (with ε=0.3 random selection)

**Reward Calculation:**
```python
total_reward = (
    hre_weight * hre_reward +
    pre_weight * accumulated_pre +
    length_penalty +
    direction_bonus
)
```

### Student Model Training

**GRPO Loss:**
```python
advantages = normalize(rewards - baseline)
policy_loss = -log_probs * clipped_advantages
kl_penalty = KL(student || reference)
loss = policy_loss + kl_weight * kl_penalty
```

**Reference Model Updates:**
- Deep copy of student weights every `update_ref_every` steps
- Gradients disabled for reference model
- Used only for KL divergence calculation

**Gradient Management:**
- Clipping at max_norm=1.0 (prevents exploding gradients)
- Accumulation over multiple trajectories
- 8-bit AdamW optimizer for memory efficiency

### Teacher Ensemble

**Model Selection:**
- Tier-based routing (free vs production)
- Dynamic weight updating based on reward feedback
- Fallback to alternative models on failure

**Batched Inference:**
- Parallel generation for right/wrong branches
- vLLM PagedAttention for memory efficiency
- Temperature=0.7 for diversity

### Reward System Details

**HRE (Hard Reward Estimator):**
```python
def calculate_reward(question, response, terminal_only=True):
    if not terminal_only or is_terminal(response):
        predicted = extract_answer(response)
        return 1.0 if predicted == reference else 0.0
    return 0.0
```

**PRE (Perplexity Reward Estimator):**
```python
def calculate_reward(question, response, normalize=True):
    log_prob = student_model.compute_log_prob(response)
    perplexity = exp(-log_prob / len(response))
    return -log(perplexity) if normalize else perplexity
```

**Accumulated PRE:**
```python
accumulated_pre = sum(
    (discount ** i) * step_pre_i
    for i, step_pre_i in enumerate(path)
)
```

### Memory Management

**Optimizations Applied:**
- 4-bit quantization (BitsAndBytes) for student model
- LoRA adapters only (freeze base model parameters)
- Gradient checkpointing (trades compute for memory)
- bf16 mixed precision on H100
- Periodic `torch.cuda.empty_cache()` calls

**Typical VRAM Usage (H100):**
- Student model: ~6-8 GB
- Teacher ensemble (vLLM): ~20-30 GB
- Training buffers: ~10-15 GB
- Total: ~40-50 GB peak

---

## Key Features

1. **Verifiers Framework Integration**:
   - Extends `MultiTurnEnv` for multi-turn reasoning tasks
   - Rubric-based reward composition with parallel scoring support
   - Standardized dataset schema with prompt/answer/info structure
   - Environment registration via `verifiers.load_environment()`

2. **vLLM Integration**:
   - High-throughput inference for teacher ensemble (DeepSeek-R1-Distill, Qwen2.5-72B, Llama-3.3-70B)
   - Terminal state checker using Nemotron-Nano-9B
   - Batched generation for efficient MCTS expansion

3. **Unsloth Optimization**:
   - Memory-efficient LoRA fine-tuning with 4-bit quantization
   - Student model: Llama-3.2-3B (changed from Qwen2.5-3B)
   - Gradient checkpointing for reduced VRAM usage
   - Flash Attention 2 for faster training

4. **GRPO Training**:
   - Group-based advantage normalization
   - KL divergence penalty with periodic reference model updates
   - Configurable β, clipping, and advantage normalization
   - Legacy mode support for ablation studies

5. **Binary Tree Exploration**:
   - Each node spawns children via similarity-based selection
   - "Right" vs "wrong" branch classification using teacher outputs
   - Binary codes track exploration paths (1=right, 0=wrong)
   - Epsilon-greedy selection for exploration-exploitation balance

6. **Advanced Reward System**:
   - Accumulated perplexity with discount factor
   - Terminal-only HRE evaluation (avoids partial response penalties)
   - Length penalties and direction bonuses
   - Configurable HRE/PRE weight balance

7. **Adaptive Compute Scaling**:
   - Environment variable control: `MAX_TREE_DEPTH`, `NUM_MCTS_ITERATIONS`, `MAX_TOKENS`
   - Dynamic dataset size configuration
   - Checkpoint saving at configurable intervals

8. **Checkpointing & Logging**:
   - Save/resume training with model snapshots every N examples
   - Accuracy and reward tracking
   - VRAM usage monitoring
   - Optional WandB integration (configurable in training.yaml)

---

## Usage

### Setup

**Prerequisites:**
- Python 3.11+
- CUDA-capable GPU (tested on H100)
- OpenRouter API key (for teacher models)

**Installation:**
```bash
cd environments/alphazero_llm_trainer

# Option 1: Using uv (recommended for Prime Intellect)
uv pip install -e .

# Option 2: Using pip
pip install -r requirements.txt

# Set API key
export OPENROUTER_API_KEY="your_key_here"
```

### Training

**Basic Usage:**
```bash
python train_vllm.py \
  --num-examples 50 \
  --checkpoint-dir ./checkpoints \
  --save-every 25 \
  --use-grpo
```

**Using Verifiers Environment:**
```python
import verifiers as vf

# Load environment
env = vf.load_environment(
    "alphazero-llm-trainer",
    tier="free",  # or "production"
    use_student_model=True
)

# Access dataset
dataset = env.dataset
print(f"Dataset size: {len(dataset)}")

# Access rubric
rubric = env.rubric
print(f"Reward functions: {len(rubric.reward_funcs)}")
print(f"Weights: {rubric.reward_weights}")
```

**Environment Variables for Scaling:**
```bash
# Adaptive compute scaling
export MAX_TREE_DEPTH=12        # Default from config: 12
export NUM_MCTS_ITERATIONS=60   # Default from config: 60
export MAX_TOKENS=1000          # Optional token budget

python train_vllm.py --num-examples 100
```

**Command Line Arguments:**
- `--num-examples`: Number of training examples (overrides config)
- `--checkpoint-dir`: Directory for model checkpoints
- `--save-every`: Checkpoint save interval
- `--log-interval`: Progress logging frequency
- `--use-grpo`: Enable GRPO training (default: uses config setting)
- `--use-legacy`: Use legacy reward-weighted supervised learning

### Evaluation

**Run Tests:**
```bash
# Test configuration loading
python test_config_only.py

# Test environment and models
python test.py

# Test GSM8K specific functionality
python test_gsm8k.py
```

**Generate Inference:**
```python
from models import StudentModel

student = StudentModel()
response = student.generate(
    prompt="What is 15 * 24?",
    max_new_tokens=256,
    temperature=0.7
)
print(response)
```

---

## Configuration

Configuration is split across multiple YAML files in `config/`:

### 1. `training.yaml` - Training Hyperparameters

**MCTS Parameters:**
```yaml
mcts:
  num_iterations: 60              # MCTS iterations per example
  exploration_constant: 1.414     # UCT exploration (sqrt(2))
  max_tree_depth: 12              # Maximum reasoning depth
  right_tokens: 10                # Tokens per "right" branch
  wrong_tokens: 10                # Tokens per "wrong" branch
```

**Reward System:**
```yaml
rewards:
  hre_weight: 0.4                 # Hard reward weight
  pre_weight: 0.6                 # Perplexity reward weight
  pre_accumulation: true          # Enable accumulated PRE
  pre_discount_factor: 0.95       # Discount for PRE accumulation
```

**GRPO Settings:**
```yaml
student_training:
  grpo:
    enabled: true
    beta: 0.1                     # Advantage scaling
    normalize_advantages: true
    clip_advantages: 3.0          # Clip outliers
    use_legacy_loss: false        # Use legacy SL mode

  kl_penalty:
    enabled: true
    kl_weight: 0.01               # KL divergence weight
    update_ref_every: 50          # Reference model update frequency
```

**Training Loop:**
```yaml
training:
  batch_size: 8
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  max_grad_norm: 1.0
  optimizer:
    type: "adamw_8bit"            # Memory-efficient optimizer
```

### 2. `models.yaml` - Model Configurations

**Student Model:**
```yaml
student_model:
  name: "unsloth/llama-3.2-3b-bnb-4bit"
  max_seq_length: 2048
  load_in_4bit: true

  lora_config:
    r: 32                         # LoRA rank
    lora_alpha: 32
    target_modules: ["q_proj", "k_proj", "v_proj", ...]
    lora_dropout: 0.05
    use_gradient_checkpointing: true
```

**Terminal Checker:**
```yaml
terminal_checker:
  name: "nvidia/nemotron-nano-9b-v2:free"
  temperature: 0.0
  max_tokens: 50
```

### 3. `teacher_models.yaml` - Teacher Ensemble

**Free Tier Models:**
```yaml
free:
  - name: "deepseek/deepseek-r1-distill-qwen-32b:free"
    max_tokens: 500
    temperature: 0.7
  - name: "nvidia/nemotron-nano-9b-v2:free"
    max_tokens: 500
    temperature: 0.7
```

**Production Tier Models:**
```yaml
production:
  - name: "qwen/qwen-2.5-72b-instruct"
    max_tokens: 500
    temperature: 0.7
  - name: "meta-llama/llama-3.3-70b-instruct"
    max_tokens: 500
    temperature: 0.7
```

### Tuning Recommendations

**For Faster Training (10-hour budget):**
- Reduce `num_mcts_iterations` to 40-60
- Lower `max_tree_depth` to 8-12
- Use `num_epochs: 1`
- Increase `save_interval` to reduce I/O

**For Better Accuracy:**
- Increase `num_mcts_iterations` to 100+
- Use production tier teachers
- Increase `pre_weight` to 0.7-0.8
- Add more teacher models to ensemble

**For Memory-Constrained GPUs:**
- Enable `gradient_checkpointing: true`
- Use `use_fp16: true` or `use_bf16: true`
- Reduce `batch_size` and increase `gradient_accumulation_steps`

---

## Related Papers & Resources

### Core Algorithms
- **AlphaZero**: [Silver et al., 2017 - Mastering Chess and Shogi by Self-Play with a General RL Algorithm](https://arxiv.org/abs/1712.01815)
- **MCTS Survey**: [Browne et al., 2012 - A Survey of Monte Carlo Tree Search Methods](https://ieeexplore.ieee.org/document/6145622)
- **UCT Algorithm**: [Kocsis & Szepesvári, 2006 - Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29)

### Reinforcement Learning
- **PPO**: [Schulman et al., 2017 - Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- **REINFORCE**: [Williams, 1992 - Simple Statistical Gradient-Following Algorithms](https://link.springer.com/article/10.1007/BF00992696)
- **RLHF**: [Christiano et al., 2017 - Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741)

### Knowledge Distillation
- **Teacher Ensemble**: [Hinton et al., 2015 - Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- **Self-Distillation**: [Furlanello et al., 2018 - Born Again Neural Networks](https://arxiv.org/abs/1805.04770)

### Math Reasoning & Datasets
- **GSM8K Dataset**: [Cobbe et al., 2021 - Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- **Chain-of-Thought**: [Wei et al., 2022 - Chain-of-Thought Prompting Elicits Reasoning in LLMs](https://arxiv.org/abs/2201.11903)
- **Process Reward Models**: [Lightman et al., 2023 - Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)

### LLM Training & Optimization
- **LoRA**: [Hu et al., 2021 - Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **QLoRA**: [Dettmers et al., 2023 - Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- **Flash Attention**: [Dao et al., 2022 - FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)

### Frameworks & Tools
- **Verifiers**: [Prime Intellect Verifiers Framework](https://github.com/PrimeIntellect-ai/verifiers) - RL environment for LLM evaluation
- **vLLM**: [Kwon et al., 2023 - Efficient Memory Management for Large Language Model Serving](https://arxiv.org/abs/2309.06180)
- **Unsloth**: [Unsloth AI - 2x faster, 50% less memory LLM finetuning](https://github.com/unslothai/unsloth)

---

## Results

### Actual Training Results

**Latest Training Run (Minimal Configuration):**

```bash
MAX_TREE_DEPTH=30 NUM_MCTS_ITERATIONS=100 python train_vllm.py \
  --num-examples 10 \
  --eval-size 1319 \
  --checkpoint-dir ./checkpoints/a100_depth30_10examples \
  --save-every 5 \
  --log-interval 2
```

**Training Configuration:**
- Training examples: 10
- Tree depth: 30
- MCTS iterations: 100 per example
- Teacher ensemble: 4 models (10% GPU each)
  - Qwen2.5-7B-Instruct (4-bit quantized)
  - Meta-Llama-3.1-8B-Instruct (4-bit quantized)
  - Mistral-7B-Instruct-v0.3 (4-bit quantized)
  - Gemma-2-9B-IT (4-bit quantized)
- Terminal checker: Qwen2.5-0.5B-Instruct (5% GPU, max_len=512)
- Hardware: A100 GPU

**Results:**

```
[TRAINED] Final Accuracy: 14.63% (193/1319)

===================================================
COMPARISON SUMMARY
===================================================
Base Model Accuracy:    12.89%
Trained Model Accuracy: 14.63%
Improvement:            +1.74%
===================================================
```

**Key Metrics:**
- Base model: 12.89% accuracy on GSM8K
- Trained model: 14.63% accuracy on GSM8K
- Absolute improvement: +1.74 percentage points
- Relative improvement: +13.5% over base model
- Evaluation set: 1,319 problems
- Total training examples: 10 only

**Notes:**
- This is a minimal proof-of-concept run with only 10 training examples
- Demonstrates +1.74% improvement even with minimal training data
- 4 quantized teacher models (BitsAndBytes 4-bit) with vLLM backend
- Efficient GPU utilization: 40% (teachers) + 5% (checker) = 45% total

**Detailed results available in:**
- `environments/alphazero_llm_trainer/gsm8k_comparison_results.json`

### Performance Benchmarks

**Expected Training Metrics (50 examples on H100):**
- Average tree size: ~30-50 nodes per example
- MCTS iterations: 60 per example
- Training time: ~2-3 hours
- Peak VRAM: ~40-50 GB
- Expected accuracy: ~40-50% on GSM8K subset (with full training)

**Optimization Tips:**
1. Use `bf16` precision on H100 (better than `fp16` for stability)
2. Enable gradient checkpointing for VRAM savings (~30% reduction)
3. Batch teacher inference in vLLM for 3-5x speedup
4. Update reference model every 50-100 steps to balance KL penalty
5. Use free tier teachers for development, production tier for final training

---

## Contributing

This is a backup repository for Prime Intellect development. For contributions:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description
4. Ensure all tests pass (`python test.py`)

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{alphazero_llm_trainer,
  title = {AlphaZero LLM Trainer: MCTS-based RL for Language Model Training},
  author = {Pradheep, Vidhya Prakash},
  year = {2025},
  url = {https://github.com/Mantissagithub/alphazero-llm-trainer},
  note = {Integrated with Prime Intellect Verifiers Framework}
}
```

---

## License

See individual component licenses. This backup is for Prime Intellect development tracking.

**Key Dependencies:**
- Verifiers: Apache 2.0
- vLLM: Apache 2.0
- Unsloth: Apache 2.0
- PyTorch: BSD-style
- Transformers: Apache 2.0
