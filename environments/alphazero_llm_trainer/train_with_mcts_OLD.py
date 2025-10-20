#!/usr/bin/env python3
"""
Training script for AlphaZero LLM Trainer with MCTS.

This script orchestrates the training loop by:
1. Loading the Verifiers environment (for dataset and rubric)
2. Running MCTS externally to generate high-quality trajectories
3. Training the student model on successful trajectories

Architecture:
- Environment: Provides dataset and evaluation (via Rubric)
- MCTS: Explores reasoning paths and identifies high-reward trajectories
- Student Model: Learns from successful trajectories to improve policy

Usage:
    python train_with_mcts.py --tier free --use-student-model --num-examples 100
"""

import os
import argparse
from typing import Dict, List
from pathlib import Path

import verifiers as vf
from openai import OpenAI

from core.mcts import MCTSSystem
from models import TeacherEnsemble, StudentModel, TerminalChecker
from rewards import CombinedRewardSystem, HardRewardEstimator, PerplexityRewardEstimator
from utils import normalize_answer
from config import get_training_config
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train AlphaZero LLM with MCTS')
    parser.add_argument(
        '--tier',
        type=str,
        default='free',
        choices=['free', 'premium'],
        help='OpenRouter API tier'
    )
    parser.add_argument(
        '--use-student-model',
        action='store_true',
        help='Use student model for PRE rewards'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=None,
        help='Number of training examples (None = use all from config)'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory to save model checkpoints'
    )
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='Log progress every N examples'
    )
    return parser.parse_args()


def initialize_components(tier: str, use_student_model: bool):
    """Initialize all components needed for training"""
    # COMMENTED OUT: OpenRouter API key check
    # api_key = os.environ.get("OPENROUTER_API_KEY")
    # if not api_key:
    #     raise ValueError("OPENROUTER_API_KEY not found in environment")

    # # Initialize OpenRouter client
    # client = OpenAI(
    #     base_url="https://openrouter.ai/api/v1",
    #     api_key=api_key
    # )

    client = None  # Placeholder since OpenRouter is not being used
    # Initialize models
    teacher_ensemble = TeacherEnsemble(client, tier=tier)
    terminal_checker = TerminalChecker(client)

    student_model = None
    if use_student_model:
        student_model = StudentModel()

    return client, teacher_ensemble, student_model, terminal_checker


def train_on_trajectories(student_model: StudentModel, trajectories: List[Dict], reward_threshold: float = 0.5):
    """Train student model on high-reward trajectories"""
    if student_model is None:
        return

    high_reward = [t for t in trajectories if t.get("reward", 0) > reward_threshold]

    if not high_reward:
        print(f"  No high-reward trajectories found (threshold: {reward_threshold})")
        return

    print(f"  Training on {len(high_reward)} high-reward trajectories...")

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
    print(f"  Average loss: {avg_loss:.4f}")


def main():
    """Main training loop"""
    args = parse_args()

    print("=" * 80)
    print("AlphaZero LLM Trainer with MCTS")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  - Tier: {args.tier}")
    print(f"  - Use Student Model: {args.use_student_model}")
    print(f"  - Num Examples: {args.num_examples or 'All (from config)'}")
    print(f"  - Checkpoint Dir: {args.checkpoint_dir}")
    print("=" * 80)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Load environment from Verifiers
    print("\nLoading environment...")
    vf_env = vf.load_environment(
        "alphazero-llm-trainer",
        tier=args.tier,
        use_student_model=args.use_student_model
    )
    print(f"Loaded environment with {len(vf_env.dataset)} examples")

    # Initialize components
    print("\nInitializing components...")
    client, teacher_ensemble, student_model, terminal_checker = initialize_components(
        args.tier,
        args.use_student_model
    )
    print("Components initialized successfully")

    # Determine number of examples to process
    num_examples = args.num_examples or len(vf_env.dataset)
    dataset_subset = vf_env.dataset.select(range(min(num_examples, len(vf_env.dataset))))

    print(f"\nStarting training on {len(dataset_subset)} examples...")
    print("=" * 80)

    # Training loop
    total_correct = 0
    total_reward = 0.0

    for idx, example in enumerate(dataset_subset):
        question = example['question']
        reference_answer = example['answer']

        print(f"\n[{idx + 1}/{len(dataset_subset)}] Processing example...")
        print(f"Question: {question[:100]}..." if len(question) > 100 else f"Question: {question}")

        # Setup reward system for this example
        correct_answer = normalize_answer(reference_answer)
        hre = HardRewardEstimator(terminal_checker, correct_answer)

        if student_model:
            pre = PerplexityRewardEstimator(student_model, terminal_checker)
            reward_system = CombinedRewardSystem(hre, pre)
        else:
            reward_system = hre

        # Initialize MCTS
        mcts = MCTSSystem(
            teacher_ensemble=teacher_ensemble,
            student_model=student_model,
            terminal_checker=terminal_checker,
            reward_system=reward_system
        )

        # Run MCTS search
        print("  Running MCTS search...")
        result = mcts.search(question, reference_answer)

        # Collect trajectories for training
        trajectories = mcts.collect_trajectories(question, reference_answer)

        # Log results
        best_reward = result.get("reward", 0.0)
        total_reward += best_reward
        if best_reward > 0.9:  # Consider correct if reward > 0.9
            total_correct += 1

        print(f"  Best reward: {best_reward:.3f}")
        print(f"  Tree size: {result.get('tree_size', 0)}")
        print(f"  Trajectories collected: {len(trajectories)}")

        # Train student model on trajectories
        if student_model and trajectories:
            train_on_trajectories(student_model, trajectories)

        # Log progress
        if (idx + 1) % args.log_interval == 0:
            avg_reward = total_reward / (idx + 1)
            accuracy = total_correct / (idx + 1)
            print("\n" + "=" * 80)
            print(f"Progress Report [{idx + 1}/{len(dataset_subset)}]")
            print(f"  - Average Reward: {avg_reward:.3f}")
            print(f"  - Accuracy: {accuracy:.2%}")
            print("=" * 80)

        # Save checkpoint
        if student_model and (idx + 1) % 50 == 0:
            checkpoint_path = checkpoint_dir / f"student_model_step_{idx + 1}.pt"
            student_model.save_checkpoint(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # Final report
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Final Statistics:")
    print(f"  - Total Examples: {len(dataset_subset)}")
    print(f"  - Total Correct: {total_correct}")
    print(f"  - Accuracy: {total_correct / len(dataset_subset):.2%}")
    print(f"  - Average Reward: {total_reward / len(dataset_subset):.3f}")
    print("=" * 80)

    # Save final model
    if student_model:
        final_path = checkpoint_dir / "student_model_final.pt"
        student_model.save_checkpoint(final_path)
        print(f"Saved final model: {final_path}")


if __name__ == "__main__":
    main()
