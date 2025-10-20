"""
rubric-compatible reward functions for verifiers framework.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI

try:
    from .hre import HardRewardEstimator
    from .pre import PerplexityRewardEstimator
    from models import TerminalChecker, StudentModel
    from utils import normalize_answer
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rewards.hre import HardRewardEstimator
    from rewards.pre import PerplexityRewardEstimator
    from models import TerminalChecker, StudentModel
    from utils import normalize_answer


def hre_reward_function(
    prompt: List[Dict],
    completion: List[Dict],
    answer: str,
    info: Dict,
    **kwargs
) -> float:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    terminal_checker = TerminalChecker(client)
    correct_answer = normalize_answer(answer)
    hre = HardRewardEstimator(terminal_checker, correct_answer)

    question_text = prompt[0]['content'] if prompt else ""
    response_text = completion[-1]['content'] if completion else ""

    return hre.calculate_reward(question_text, response_text, terminal_only=True)


def pre_reward_function(
    prompt: List[Dict],
    completion: List[Dict],
    state: Dict,
    **kwargs
) -> float:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    try:
        student = StudentModel()
        terminal_checker = TerminalChecker(client)
        pre = PerplexityRewardEstimator(student, terminal_checker)
    except Exception as e:
        return 0.0

    question_text = prompt[0]['content'] if prompt else ""
    response_text = completion[-1]['content'] if completion else ""

    return pre.calculate_reward(question_text, response_text, normalize=True)


def combined_reward_function(
    prompt: List[Dict],
    completion: List[Dict],
    answer: str,
    info: Dict,
    state: Dict,
    hre_weight: float = 1.0,
    pre_weight: float = 0.6,
    **kwargs
) -> float:
    hre = hre_reward_function(prompt, completion, answer, info, **kwargs)
    pre = pre_reward_function(prompt, completion, state, **kwargs)

    total_weight = hre_weight + pre_weight
    norm_hre_weight = hre_weight / total_weight
    norm_pre_weight = pre_weight / total_weight

    return norm_hre_weight * hre + norm_pre_weight * pre


__all__ = [
    'hre_reward_function',
    'pre_reward_function',
    'combined_reward_function'
]
