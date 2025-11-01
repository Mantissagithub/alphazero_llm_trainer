# alphazero-llm-trainer

mcts-guided training environment for llms on mathematical reasoning with adaptive depth control

## overview

- **environment id**: `alphazero-llm-trainer`
- **task type**: multi-turn mcts tree search
- **dataset**: gsm8k (openai/gsm8k)
- **base class**: verifiers `MultiTurnEnv` with adaptive termination
- **tags**: math, reasoning, mcts, grpo, eval, train

## quickstart

### evaluate with vf-eval

```bash
# default evaluation (uses depth=12 from config)
uv run vf-eval alphazero-llm-trainer -m gpt-4o-mini -n 5

# laptop eval with shallow depth (faster)
MAX_TREE_DEPTH=5 uv run vf-eval alphazero-llm-trainer -m gpt-4o-mini -n 5

# deep search with more compute
MAX_TREE_DEPTH=50 NUM_MCTS_ITERATIONS=200 uv run vf-eval alphazero-llm-trainer -m gpt-4o-mini -n 10
```

### train the student model

```bash
# train with mcts + grpo (default settings)
python train_vllm.py --num-examples 50

# train with custom depth
MAX_TREE_DEPTH=20 python train_vllm.py --num-examples 100 --use-grpo

# train with adaptive depth and token budget
MAX_TREE_DEPTH=15 MAX_TOKENS=5000 python train_vllm.py --num-examples 200
```

## key feature: adaptive depth control

**what changed:** the environment now uses `MultiTurnEnv` with adaptive depth control via environment variables instead of hardcoded config values.

**why:** allows scaling mcts search depth based on available hardware without modifying config files.

### depth scaling by hardware

| hardware | recommended depth | iterations | expected time |
|----------|------------------|------------|---------------|
| laptop / cpu | 5 | 30-40 | 2-5 min/sample |
| single gpu | 12 (default) | 60 | 5-10 min/sample |
| multi-gpu / a100 | 20-30 | 100-150 | 10-15 min/sample |
| h100 cluster | 50+ | 200+ | 15-20 min/sample |

### environment variables

| variable | default | description |
|----------|---------|-------------|
| `MAX_TREE_DEPTH` | 12 (from config) | maximum tree depth - overrides config |
| `NUM_MCTS_ITERATIONS` | 60 (from config) | iterations per search - overrides config |
| `MAX_TOKENS` | 0 (unlimited) | optional token budget for cost control |

**important:** these env vars override config values, enabling compute-adaptive scaling without editing yaml files.

### usage examples

```bash
# quick laptop test (shallow)
MAX_TREE_DEPTH=5 python train_vllm.py --num-examples 10

# standard workstation training
MAX_TREE_DEPTH=12 python train_vllm.py --num-examples 50

# deep search on h100
MAX_TREE_DEPTH=50 NUM_MCTS_ITERATIONS=200 python train_vllm.py --num-examples 1000

# cost-controlled with token budget
MAX_TREE_DEPTH=15 MAX_TOKENS=3000 python train_vllm.py --num-examples 100
```

## command reference

### vf-eval commands

```bash
# basic evaluation
uv run vf-eval alphazero-llm-trainer -m <model> -n <samples>

# with adaptive depth
MAX_TREE_DEPTH=<depth> uv run vf-eval alphazero-llm-trainer -m <model> -n <samples>

# multiple rollouts
uv run vf-eval alphazero-llm-trainer -m <model> -n <samples> -r <rollouts>
```

### training commands

```bash
# basic training
python train_vllm.py --num-examples <n>

# with adaptive depth
MAX_TREE_DEPTH=<depth> python train_vllm.py --num-examples <n>

# with all options
MAX_TREE_DEPTH=<depth> NUM_MCTS_ITERATIONS=<iters> MAX_TOKENS=<budget> \
  python train_vllm.py --num-examples <n> --use-grpo --save-every <freq>
```

### environment arguments (load_environment)

| argument | type | default | description |
|----------|------|---------|-------------|
| `tier` | str | `"free"` | model tier for teacher ensemble |
| `use_student_model` | bool | `false` | enable student model for perplexity rewards |

### training script arguments (train_vllm.py)

| argument | type | default | description |
|----------|------|---------|-------------|
| `--num-examples` | int | 50 (config) | number of training examples |
| `--checkpoint-dir` | str | ./checkpoints | checkpoint save directory |
| `--save-every` | int | 300 (config) | checkpoint save frequency |
| `--log-interval` | int | 10 (config) | logging frequency |
| `--use-grpo` | flag | config | enable grpo training mode |
| `--use-legacy` | flag | false | use legacy supervised learning |

## metrics

| metric | description | range |
|--------|-------------|-------|
| `reward` | combined weighted reward (0.4*hre + 0.6*pre) | 0.0 - 1.0 |
| `correctness_reward` | binary correctness from hre | 0.0 or 1.0 |
| `perplexity_reward` | trajectory quality from pre (if enabled) | 0.0 - 1.0 |
| `depth` | actual tree depth reached | 0 - MAX_TREE_DEPTH |
| `terminal` | whether problem was solved | true/false |

## implementation details

### what was changed

**1. singleturnenv → multiturnenv**
- converted from `vf.SingleTurnEnv` to `vf.MultiTurnEnv`
- enables multi-turn mcts tree search with proper state management
- implements `env_response()` (returns empty string for single-agent)
- implements `is_completed()` with multi-condition termination

**2. adaptive depth via environment variables**
- `MAX_TREE_DEPTH` overrides config for compute-adaptive scaling
- no hardcoded depth limits from config files
- scales from 5 (laptop) to 50+ (h100 cluster)

**3. proper vllm backend**
- replaced placeholder terminal checker with `VLLMTerminalChecker`
- uses qwen 2.5 0.5b for fast terminal state detection
- proper vllm initialization for efficient inference

**4. verifiers-compatible dataset schema**
- changed dataset column from `'question'` to `'prompt'`
- matches verifiers expected schema for seamless integration
- format: `{'prompt': question, 'answer': solution, 'info': {...}}`

**5. multi-condition termination**
- safety limit: turn >= 100 (base class)
- adaptive depth: depth >= MAX_TREE_DEPTH
- terminal state: problem solved
- mcts exhausted: no more promising branches
- token budget: total_tokens >= MAX_TOKENS (if configured)

### mcts search system

- monte carlo tree search with adaptive depth (5-50+ levels)
- ucb1 selection with exploration constant √2
- binary expansion (right/wrong token branches)
- terminates when any condition above is met

### reward system

**hard reward estimator (hre) - 40% weight:**
- validates numerical answers against ground truth
- uses vllm terminal checker for fast inference
- binary reward: 1.0 if correct, 0.0 otherwise

**perplexity reward estimator (pre) - 60% weight:**
- measures trajectory quality via student model perplexity
- normalized against baseline perplexity
- only active if `use_student_model=True`

**combined rewards:**
- weighted combination: 0.4 * hre + 0.6 * pre
- length penalties for efficiency (-0.01 * path_length)
- direction bonuses (0.1 * right_count - 0.05 * wrong_count)

### components

- **student model**: llama 3.2 3b (4-bit quantized, lora r=32)
- **teacher ensemble**: deepseek r1, gpt-oss, llama 3.3, qwen3 (free tier)
- **terminal checker**: qwen 2.5 0.5b (vllm backend, 5% gpu memory)
- **training mode**: grpo (group relative policy optimization)

## dataset

- **source**: openai/gsm8k from huggingface
- **format**: mathematical word problems with step-by-step solutions
- **schema**: `{'prompt': question, 'answer': solution}`
- **size**: configurable (default: 50 train, 20 eval)

## configuration

all hyperparameters are in yaml config files:

- `config/training.yaml` - training hyperparameters
- `config/models.yaml` - model configurations
- `config/teacher_models.yaml` - teacher ensemble models

key settings:
- mcts iterations: 60 (override with `NUM_MCTS_ITERATIONS`)
- max tree depth: 12 (override with `MAX_TREE_DEPTH`)
- reward weights: 40% hre, 60% pre
- batch size: 8 (gradient accumulation: 4)
- learning rate: 2e-4 (warmup: 50 steps)

## complete examples

### 1. quick laptop test (shallow depth)

```bash
# evaluation
MAX_TREE_DEPTH=5 uv run vf-eval alphazero-llm-trainer -m gpt-4o-mini -n 5

# training
MAX_TREE_DEPTH=5 python train_vllm.py --num-examples 10
```

### 2. standard workstation training (default depth)

```bash
# evaluation
uv run vf-eval alphazero-llm-trainer -m gpt-4o-mini -n 10

# training with grpo
MAX_TREE_DEPTH=12 python train_vllm.py --num-examples 50 --use-grpo
```

### 3. deep search on h100 cluster

```bash
# evaluation with deep search
MAX_TREE_DEPTH=50 NUM_MCTS_ITERATIONS=200 uv run vf-eval alphazero-llm-trainer -m gpt-4o-mini -n 20

# training with maximum quality
MAX_TREE_DEPTH=50 NUM_MCTS_ITERATIONS=200 python train_vllm.py --num-examples 1000 --use-grpo
```

### 4. cost-controlled with token budget

```bash
# evaluation with token limit
MAX_TREE_DEPTH=15 MAX_TOKENS=3000 uv run vf-eval alphazero-llm-trainer -m gpt-4o-mini -n 50

# training with token limit
MAX_TREE_DEPTH=15 MAX_TOKENS=5000 python train_vllm.py --num-examples 100
```

## architecture

```
alphazero_llm_trainer/
├── alphazero_llm_trainer.py  # entry point (load_environment)
├── core/
│   ├── environment.py         # multiturnenv with adaptive depth
│   ├── mcts.py               # mcts search system
│   └── tree.py               # tree node implementation
├── models/
│   ├── student_model.py      # llama 3.2 3b with lora
│   ├── teacher_ensemble_vllm.py  # teacher ensemble
│   └── terminal_checker_vllm.py  # vllm terminal checker
├── rewards/
│   ├── hre.py                # hard reward estimator
│   ├── pre.py                # perplexity reward estimator
│   └── combined.py           # combined reward system
├── config/
│   ├── training.yaml         # training hyperparameters
│   ├── models.yaml           # model configurations
│   └── teacher_models.yaml   # teacher ensemble config
└── train_vllm.py             # training script
```

## requirements

- python >= 3.11
- verifiers >= 0.1.5.post0
- vllm >= 0.11.0
- torch == 2.8.0
- unsloth >= 2024.8
- transformers, datasets, numpy, scikit-learn

## testing

run configuration tests:

```bash
cd /home/pradheep/alphazero-llm-trainer/environments/alphazero_llm_trainer
python test_config_only.py
```

expected output:
```
✓ default depth: 12 (from config)
✓ laptop depth: 5 (from MAX_TREE_DEPTH env var)
✓ h100 depth: 50 (from MAX_TREE_DEPTH env var)
✓ mcts iterations: 200 (from NUM_MCTS_ITERATIONS env var)
✓ token budget: 1000 (from MAX_TOKENS env var)
✓ dataset schema: prompt/answer columns present
✓ is_completed: adaptive depth check works
✓ is_completed: terminal state check works
✓ is_completed: mcts exhaustion check works
✓ is_completed: token budget check works
```

## expected performance

typical results on gsm8k (may vary by hardware and model selection):

| depth | iterations | est. accuracy | time/sample | hardware |
|-------|------------|---------------|-------------|----------|
| 5 | 30-40 | ~55-65% | 2-5 min | laptop/cpu |
| 12 | 60 | ~70-80% | 5-10 min | single gpu |
| 20 | 100 | ~80-90% | 10-15 min | a100 |
| 50 | 200+ | ~85-95% | 15-20 min | h100 |

*actual performance depends on model selection, teacher quality, and training configuration

