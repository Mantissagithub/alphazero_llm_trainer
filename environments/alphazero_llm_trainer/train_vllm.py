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
        default=350,
        help='Number of training examples'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=50,
        help='Save checkpoint every N examples'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Log progress every N examples'
    )
    parser.add_argument(
        '--use-grpo',
        action='store_true',
        default=True,
        help='Use GRPO (Group Relative Policy Optimization) for training'
    )
    parser.add_argument(
        '--grpo-beta',
        type=float,
        default=0.1,
        help='GRPO KL penalty coefficient'
    )
    parser.add_argument(
        '--grpo-clip',
        type=float,
        default=3.0,
        help='GRPO advantage clipping threshold'
    )
    parser.add_argument(
        '--reward-threshold',
        type=float,
        default=0.5,
        help='Minimum reward for training trajectories (legacy mode only)'
    )
    return parser.parse_args()


def train_on_trajectories_grpo(
    student_model: StudentModel,
    trajectories: List[Dict],
    beta: float = 0.1,
    clip_advantages: float = 3.0,
    kl_weight: float = 0.01
):
    if student_model is None or not trajectories:
        return

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
        clip_advantages=clip_advantages,
        kl_weight=kl_weight
    )

    grpo_loss.backward()
    torch.nn.utils.clip_grad_norm_(student_model.model.parameters(), max_norm=1.0)
    student_model.optimizer.step()
    student_model.optimizer.zero_grad()

    student_model.prepare_for_inference()

    print(f"    grpo loss: {grpo_loss.item():.4f}")


def train_on_trajectories_legacy(
    student_model: StudentModel,
    trajectories: List[Dict],
    reward_threshold: float = 0.5
):
    if student_model is None:
        return

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

    student_model.optimizer.step()
    student_model.optimizer.zero_grad()
    student_model.prepare_for_inference()

    avg_loss = total_loss / len(high_reward)
    print(f"    loss: {avg_loss:.4f}")



def main():
    args = parse_args()
    config = get_training_config()

    print("=" * 80)
    print("alphazero llm training - vllm accelerated")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("❌ No GPU found! vLLM requires CUDA.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    print(f"Examples: {args.num_examples}")
    print(f"Training Mode: {'GRPO' if args.use_grpo else 'Legacy Reward-Weighted SL'}")
    if args.use_grpo:
        print(f"GRPO Beta: {args.grpo_beta} | Clip: {args.grpo_clip}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
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
    student_model = StudentModel()
    print("✓ student loaded")

    print("\n" + "=" * 80)
    vram_gb = torch.cuda.memory_allocated() / 1e9
    vram_reserved_gb = torch.cuda.memory_reserved() / 1e9
    print(f"VRAM allocated: {vram_gb:.2f} GB")
    print(f"VRAM reserved: {vram_reserved_gb:.2f} GB / {gpu_memory_gb:.1f} GB")
    print(f"available: {gpu_memory_gb - vram_reserved_gb:.2f} GB")
    print("=" * 80)

    num_examples = min(args.num_examples, len(env.dataset))
    dataset_subset = env.dataset.select(range(num_examples))

    kl_weight = config["student_training"]["kl_penalty"]["kl_weight"]
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
            if args.use_grpo:
                train_on_trajectories_grpo(
                    student_model,
                    trajectories,
                    beta=args.grpo_beta,
                    clip_advantages=args.grpo_clip,
                    kl_weight=kl_weight
                )
            else:
                train_on_trajectories_legacy(
                    student_model,
                    trajectories,
                    reward_threshold=args.reward_threshold
                )

        if (idx + 1) % update_ref_every == 0:
            student_model.update_ref_model()
            print(f"  🔄 reference model updated")

        if (idx + 1) % args.save_every == 0:
            ckpt = checkpoint_dir / f"student_step_{idx + 1}.pt"
            student_model.save_checkpoint(str(ckpt))
            print(f"  💾 saved: {ckpt}")

        if (idx + 1) % args.log_interval == 0:
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

    final = checkpoint_dir / "student_final.pt"
    student_model.save_checkpoint(str(final))
    print(f"\n✓ final model: {final}")

    final_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n📊 peak vram usage: {final_vram_gb:.2f} GB")


if __name__ == "__main__":
    main()
