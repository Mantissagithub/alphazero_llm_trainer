#!/usr/bin/env python3

import os
import sys
import torch
from pathlib import Path
import argparse
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent / "environments" / "alphazero_llm_trainer"))

import verifiers as vf
from core.mcts import MCTSSystem
from models import TeacherEnsemble, StudentModel, TerminalChecker
from rewards import HardRewardEstimator, PerplexityRewardEstimator, CombinedRewardSystem
from utils import normalize_answer
from config import get_training_config


def parse_args():
    parser = argparse.ArgumentParser(description='train alphazero llm with vllm')
    parser.add_argument(
        '--num-examples',
        type=int,
        default=None,
        help='Number of training examples (overrides config)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save model checkpoints (overrides config)'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=None,
        help='Save checkpoint every N examples (overrides config)'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=None,
        help='Log progress every N examples (overrides config)'
    )
    parser.add_argument(
        '--use-grpo',
        action='store_true',
        default=None,
        help='Use GRPO (overrides config)'
    )
    parser.add_argument(
        '--use-legacy',
        action='store_true',
        default=False,
        help='Use legacy training mode (overrides config)'
    )
    return parser.parse_args()


def train_on_trajectories_grpo(
    student_model: StudentModel,
    trajectories: List[Dict],
    config: Dict
):
    if student_model is None or not trajectories:
        return

    # Get GRPO settings from config
    grpo_config = config["student_training"]["grpo"]
    beta = grpo_config["beta"]
    clip_advantages = grpo_config["clip_advantages"]
    normalize_advantages = grpo_config["normalize_advantages"]

    kl_config = config["student_training"]["kl_penalty"]
    kl_weight = kl_config["kl_weight"]

    print(f"  ✓ training with grpo on {len(trajectories)} trajectories")

    student_model.prepare_for_training()

    grouped_trajs = []
    for traj in trajectories:
        grouped_trajs.append({
            'text': f"{traj.get('question', '')}\n{traj.get('next_state', '')}",
            'reward': traj.get('reward', 0.0)
        })

    grpo_loss = student_model.compute_grpo_loss(
        grouped_trajs,
        beta=beta,
        clip_advantages=clip_advantages if normalize_advantages else 0.0,
        kl_weight=kl_weight
    )

    grpo_loss.backward()

    # Use max_grad_norm from config
    max_grad_norm = config["training"]["max_grad_norm"]
    torch.nn.utils.clip_grad_norm_(student_model.model.parameters(), max_norm=max_grad_norm)

    student_model.optimizer.step()
    student_model.optimizer.zero_grad()

    student_model.prepare_for_inference()

    print(f"    grpo loss: {grpo_loss.item():.4f}")


def train_on_trajectories_legacy(
    student_model: StudentModel,
    trajectories: List[Dict],
    config: Dict
):
    if student_model is None:
        return

    # Get reward threshold from config
    reward_threshold = config["student_training"]["min_reward_threshold"]

    high_reward = [t for t in trajectories if t.get("reward", 0) > reward_threshold]

    if not high_reward:
        return

    print(f"  ✓ training (legacy) on {len(high_reward)} high-reward trajectories")

    student_model.prepare_for_training()

    total_loss = 0.0
    for traj in high_reward:
        question = traj.get('question', '')
        next_state = traj.get('next_state', '')
        reward = traj.get('reward', 0)

        text = f"{question}\n{next_state}"
        loss = student_model.compute_loss(text, reward)
        loss.backward()
        total_loss += loss.item()

    # Use max_grad_norm from config
    max_grad_norm = config["training"]["max_grad_norm"]
    torch.nn.utils.clip_grad_norm_(student_model.model.parameters(), max_norm=max_grad_norm)

    student_model.optimizer.step()
    student_model.optimizer.zero_grad()
    student_model.prepare_for_inference()

    avg_loss = total_loss / len(high_reward)
    print(f"    loss: {avg_loss:.4f}")



def main():
    args = parse_args()
    config = get_training_config()

    # Use config values with optional arg overrides
    num_examples = args.num_examples if args.num_examples is not None else config["dataset"]["train_size"]
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else config["checkpointing"]["save_dir"]
    save_interval = args.save_every if args.save_every is not None else config["logging"]["save_interval"]
    log_interval = args.log_interval if args.log_interval is not None else config["logging"]["log_interval"]

    # Determine training mode from config or args
    if args.use_legacy:
        use_grpo = False
    elif args.use_grpo is not None:
        use_grpo = args.use_grpo
    else:
        use_grpo = config["student_training"]["grpo"]["enabled"]

    print("=" * 80)
    print("alphazero llm training - vllm accelerated")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ No GPU found! vLLM requires CUDA.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    print(f"Examples: {num_examples}")
    print(f"Training Mode: {'GRPO' if use_grpo else 'Legacy Reward-Weighted SL'}")
    if use_grpo:
        grpo_config = config["student_training"]["grpo"]
        print(f"GRPO Beta: {grpo_config['beta']} | Clip: {grpo_config['clip_advantages']}")
        print(f"Normalize Advantages: {grpo_config['normalize_advantages']}")
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Save Interval: {save_interval} | Log Interval: {log_interval}")
    print(f"Learning Rate: {config['student_training']['learning_rate']}")
    print(f"Max Grad Norm: {config['training']['max_grad_norm']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print("=" * 80)

    print("\nloading verifiers environment...")
    env = vf.load_environment("alphazero-llm-trainer", use_student_model=True)
    print(f"✓ dataset: {len(env.dataset)} examples")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("initializing models (vllm)")
    print("=" * 80)

    print("\n[1/3] teacher ensemble (vllm)...")
    teacher_ensemble = TeacherEnsemble(device="cuda:0")

    print("\n[2/3] terminal checker (vllm)...")
    terminal_checker = TerminalChecker(device="cuda:0")

    print("\n[3/3] student model (unsloth)...")
    student_model = StudentModel(learning_rate=config["student_training"]["learning_rate"])
    print("✓ student loaded")

    print("\n" + "=" * 80)
    vram_gb = torch.cuda.memory_allocated() / 1e9
    vram_reserved_gb = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM allocated: {vram_gb:.2f} GB")
    print(f"VRAM reserved: {vram_reserved_gb:.2f} GB / {gpu_memory_gb:.1f} GB")
    print(f"available: {gpu_memory_gb - vram_reserved_gb:.2f} GB")
    print("=" * 80)

    dataset_subset = env.dataset.select(range(num_examples))

    update_ref_every = config["student_training"]["kl_penalty"]["update_ref_every"]

    total_correct = 0
    total_reward = 0.0

    print("\n" + "=" * 80)
    print(f"starting training ({len(dataset_subset)} examples)")
    print("=" * 80)

    for idx, example in enumerate(dataset_subset):
        question = example['question']
        reference = example['answer']

        print(f"\n[{idx + 1}/{len(dataset_subset)}] {question[:70]}...")

        correct_answer = normalize_answer(reference)
        hre = HardRewardEstimator(terminal_checker, correct_answer)
        pre = PerplexityRewardEstimator(student_model, terminal_checker)
        reward_system = CombinedRewardSystem(hre, pre)

        mcts = MCTSSystem(
            teacher_ensemble=teacher_ensemble,
            student_model=student_model,
            terminal_checker=terminal_checker,
            reward_system=reward_system
        )

        print("  running mcts...")
        result = mcts.search(question, reference)
        trajectories = mcts.collect_trajectories(question, reference)

        best_reward = result.get("reward", 0.0)
        total_reward += best_reward

        if best_reward > 0.9:
            total_correct += 1

        tree_size = result.get('tree_size', 0)
        print(f"  reward: {best_reward:.3f} | nodes: {tree_size} | trajectories: {len(trajectories)}")

        if trajectories:
            if use_grpo:
                train_on_trajectories_grpo(
                    student_model,
                    trajectories,
                    config
                )
            else:
                train_on_trajectories_legacy(
                    student_model,
                    trajectories,
                    config
                )

        if (idx + 1) % update_ref_every == 0:
            student_model.update_ref_model()
            print(f"  🔄 reference model updated")

        if (idx + 1) % save_interval == 0:
            ckpt = Path(checkpoint_dir) / f"student_step_{idx + 1}.pt"
            student_model.save_checkpoint(str(ckpt))
            print(f"  💾 saved: {ckpt}")

        if (idx + 1) % log_interval == 0:
            acc = total_correct / (idx + 1)
            avg_r = total_reward / (idx + 1)
            print("\n" + "-" * 80)
            print(f"📊 progress [{idx + 1}/{len(dataset_subset)}]")
            print(f"   accuracy: {acc:.2%} | avg reward: {avg_r:.3f}")
            print("-" * 80)

    accuracy = total_correct / len(dataset_subset)
    avg_reward = total_reward / len(dataset_subset)

    print("\n" + "=" * 80)
    print("training complete!")
    print("=" * 80)
    print(f"total examples: {len(dataset_subset)}")
    print(f"total correct: {total_correct}")
    print(f"accuracy: {accuracy:.2%}")
    print(f"avg reward: {avg_reward:.3f}")
    print("=" * 80)

    final = Path(checkpoint_dir) / "student_final.pt"
    student_model.save_checkpoint(str(final))
    print(f"\n✓ final model: {final}")

    final_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n📊 peak vram usage: {final_vram_gb:.2f} GB")


if __name__ == "__main__":
    main()
