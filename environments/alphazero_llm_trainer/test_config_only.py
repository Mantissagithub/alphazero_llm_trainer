#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent))


def test_adaptive_depth_config():
    # mock vllm terminal checker to avoid loading model
    with patch('core.environment.VLLMTerminalChecker') as mock_checker:
        mock_checker.return_value = Mock()

        from core.environment import AlphaZeroLLMEnvironment

        # test 1: default depth from config (12)
        if 'MAX_TREE_DEPTH' in os.environ:
            del os.environ['MAX_TREE_DEPTH']
        env = AlphaZeroLLMEnvironment()
        assert env.max_tree_depth == 12, f"default: expected 12, got {env.max_tree_depth}"
        assert env.max_turns == 100, f"safety: expected 100, got {env.max_turns}"
        print("✓ default depth: 12 (from config)")

        # test 2: laptop depth (5)
        os.environ['MAX_TREE_DEPTH'] = '5'
        env = AlphaZeroLLMEnvironment()
        assert env.max_tree_depth == 5, f"laptop: expected 5, got {env.max_tree_depth}"
        print("✓ laptop depth: 5 (from MAX_TREE_DEPTH env var)")

        # test 3: h100 cluster depth (50)
        os.environ['MAX_TREE_DEPTH'] = '50'
        env = AlphaZeroLLMEnvironment()
        assert env.max_tree_depth == 50, f"h100: expected 50, got {env.max_tree_depth}"
        print("✓ h100 depth: 50 (from MAX_TREE_DEPTH env var)")

        # test 4: mcts iterations configurable
        os.environ['NUM_MCTS_ITERATIONS'] = '200'
        env = AlphaZeroLLMEnvironment()
        assert env.num_mcts_iterations == 200, f"expected 200, got {env.num_mcts_iterations}"
        print("✓ mcts iterations: 200 (from NUM_MCTS_ITERATIONS env var)")

        # test 5: optional token budget
        os.environ['MAX_TOKENS'] = '1000'
        env = AlphaZeroLLMEnvironment()
        assert env.max_tokens == 1000, f"expected 1000, got {env.max_tokens}"
        print("✓ token budget: 1000 (from MAX_TOKENS env var)")

        # test 6: dataset schema
        os.environ['MAX_TREE_DEPTH'] = '12'  # reset
        env = AlphaZeroLLMEnvironment()
        first_item = env.dataset[0]
        assert 'prompt' in first_item, "dataset missing 'prompt'"
        assert 'answer' in first_item, "dataset missing 'answer'"
        print("✓ dataset schema: prompt/answer columns present")


async def test_is_completed_logic():
    with patch('core.environment.VLLMTerminalChecker') as mock_checker:
        mock_checker.return_value = Mock()

        from core.environment import AlphaZeroLLMEnvironment

        os.environ['MAX_TREE_DEPTH'] = '10'
        env = AlphaZeroLLMEnvironment()

        # test depth limit
        state = {'turn': 0, 'depth': 9, 'terminal': False}
        result = await env.is_completed([], state)
        assert not result, "should not complete at depth 9"

        state = {'turn': 0, 'depth': 10, 'terminal': False}
        result = await env.is_completed([], state)
        assert result, "should complete at depth 10"
        print("✓ is_completed: adaptive depth check works")

        # test terminal state
        state = {'turn': 0, 'depth': 5, 'terminal': True}
        result = await env.is_completed([], state)
        assert result, "should complete when terminal=True"
        print("✓ is_completed: terminal state check works")

        # test mcts exhaustion
        state = {'turn': 0, 'depth': 5, 'terminal': False, 'mcts_exhausted': True}
        result = await env.is_completed([], state)
        assert result, "should complete when mcts_exhausted=True"
        print("✓ is_completed: mcts exhaustion check works")

        # test token budget
        os.environ['MAX_TOKENS'] = '500'
        env = AlphaZeroLLMEnvironment()
        state = {'turn': 0, 'depth': 5, 'terminal': False, 'total_tokens': 501}
        result = await env.is_completed([], state)
        assert result, "should complete when token budget exceeded"
        print("✓ is_completed: token budget check works")


if __name__ == '__main__':
    import asyncio

    print("\n" + "="*60)
    print("testing adaptive depth configuration (mocked)")
    print("="*60 + "\n")

    test_adaptive_depth_config()

    print("\n" + "="*60)
    print("testing is_completed logic (mocked)")
    print("="*60 + "\n")

    asyncio.run(test_is_completed_logic())

    print("\n" + "="*60)
    print("✓ all configuration tests passed!")
    print("="*60 + "\n")

    print("environment is ready for:")
    print("  • MAX_TREE_DEPTH=5 uv run vf-eval alphazero-llm-trainer")
    print("  • MAX_TREE_DEPTH=12 python train_vllm.py")
    print("  • MAX_TREE_DEPTH=50 (on h100 cluster)")
    print()

