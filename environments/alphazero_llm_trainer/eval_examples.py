import os
import sys
from pathlib import Path
import asyncio
import json
from typing import Dict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import verifiers as vf
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


async def evaluate_examples(
    env_id: str = "alphazero-llm-trainer",
    model_name: str = "google/gemini-2.0-flash-001",
    num_examples: int = 10,
    rollouts_per_example: int = 1,
    env_args: Dict = None,
    sampling_args: Dict = None,
    output_file: str = None
):

    if env_args is None:
        env_args = {
            "use_student_model": False,
            "use_combined": True,
            "num_train_examples": 500,
            "num_eval_examples": num_examples
        }

    if sampling_args is None:
        sampling_args = {
            "temperature": 0.1,
            "max_tokens": None
        }

    api_key = os.environ.get("PRIME_API_KEY", "")
    if not api_key:
        raise ValueError("PRIME_API_KEY environment variable must be set")

    base_url = os.environ.get("PRIME_INFERENCE_URL", "https://api.pinference.ai/api/v1")

    print("=" * 80)
    print("AlphaZero LLM Trainer - Evaluation")
    print("=" * 80)
    print(f"Environment: {env_id}")
    print(f"Model: {model_name}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")
    print(f"API Base URL: {base_url}")
    print("=" * 80)

    print("\nLoading environment...")
    env = vf.load_environment(env_id, **env_args)
    print(f"✓ Environment loaded")
    print(f"  Train dataset size: {len(env.dataset)}")
    print(f"  Eval dataset size: {len(env.eval_dataset) if hasattr(env, 'eval_dataset') else 'N/A'}")

    dataset = env.eval_dataset if hasattr(env, 'eval_dataset') and env.eval_dataset else env.dataset
    print(f"  Using dataset size: {len(dataset)}")

    if num_examples > len(dataset):
        print(f"Warning: Requested {num_examples} examples, but dataset only has {len(dataset)}")
        num_examples = len(dataset)

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )

    print(f"\nRunning evaluation on {num_examples} examples...")
    print("-" * 80)

    results = await env.evaluate(
        client=client,
        model=model_name,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        sampling_args=sampling_args,
        print_results=True,
        verbose=True
    )

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)

    if results:
        reward_key = 'reward' if 'reward' in results else 'total_reward'
        if reward_key in results:
            rewards = results[reward_key]
            print(f"\nReward Statistics:")
            print(f"  Average: {rewards.get('mean', 0):.4f}")
            print(f"  Std Dev: {rewards.get('std', 0):.4f}")
            print(f"  Min: {rewards.get('min', 0):.4f}")
            print(f"  Max: {rewards.get('max', 0):.4f}")

        print(f"\nAll Metrics:")
        for key, value in results.items():
            if isinstance(value, dict) and 'mean' in value:
                print(f"  {key}: {value.get('mean', 0):.4f} ± {value.get('std', 0):.4f}")

    if output_file:
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "env_id": env_id,
            "model": model_name,
            "num_examples": num_examples,
            "rollouts_per_example": rollouts_per_example,
            "env_args": env_args,
            "sampling_args": sampling_args,
            "results": results
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n✓ Results saved to {output_file}")

    print("=" * 80)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate alphazero-llm-trainer environment')
    parser.add_argument('--num-examples', type=int, default=10, help='Number of examples to evaluate')
    parser.add_argument('--rollouts', type=int, default=1, help='Rollouts per example')
    parser.add_argument('--model', type=str, default='google/gemini-2.0-flash-001', help='Model name')
    parser.add_argument('--temperature', type=float, default=0.1, help='Sampling temperature')
    parser.add_argument('--max-tokens', type=int, default=None, help='Max tokens')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--use-combined', action='store_true', default=True, help='Use combined rewards')
    parser.add_argument('--use-student-model', action='store_true', default=False, help='Use student model')
    parser.add_argument('--num-train-examples', type=int, default=500, help='Number of training examples')

    args = parser.parse_args()

    env_args = {
        "use_student_model": args.use_student_model,
        "use_combined": args.use_combined,
        "num_train_examples": args.num_train_examples,
        "num_eval_examples": args.num_examples
    }

    sampling_args = {
        "temperature": args.temperature
    }

    if args.max_tokens:
        sampling_args["max_tokens"] = args.max_tokens

    asyncio.run(evaluate_examples(
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts,
        model_name=args.model,
        env_args=env_args,
        sampling_args=sampling_args,
        output_file=args.output
    ))


if __name__ == "__main__":
    main()

