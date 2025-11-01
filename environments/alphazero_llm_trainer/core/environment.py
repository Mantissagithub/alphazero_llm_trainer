import os
from typing import Dict, Optional, Any, List
import verifiers as vf
from verifiers.types import Messages, State
from datasets import Dataset

from models import StudentModel
from models.terminal_checker_vllm import VLLMTerminalChecker
from rewards import HardRewardEstimator, PerplexityRewardEstimator, CombinedRewardSystem
from utils import load_gsm8k_dataset, normalize_answer
from config import get_training_config


class AlphaZeroLLMEnvironment(vf.MultiTurnEnv):
    def __init__(
        self,
        tier: str = "free",
        use_student_model: bool = False,
        **kwargs
    ):
        self.tier = tier
        self.use_student_model = use_student_model

        # load config for defaults
        config = get_training_config()

        # adaptive depth from env var (overrides config for compute scaling)
        self.max_tree_depth = int(os.environ.get(
            'MAX_TREE_DEPTH',
            config['mcts']['max_tree_depth']
        ))

        # optional token budget from env var
        self.max_tokens = int(os.environ.get('MAX_TOKENS', 0))

        # mcts iterations configurable via env var
        self.num_mcts_iterations = int(os.environ.get(
            'NUM_MCTS_ITERATIONS',
            config['mcts']['num_iterations']
        ))

        # initialize vllm terminal checker
        self.terminal_checker = VLLMTerminalChecker()

        # student model for perplexity rewards
        self.student_model = None
        if use_student_model:
            self.student_model = StudentModel()

        # load dataset with verifiers schema (prompt/answer)
        train_size = config["dataset"]["train_size"]
        data = load_gsm8k_dataset("train", train_size)

        dataset = Dataset.from_list([
            {
                'prompt': item['question'],  # verifiers expects 'prompt'
                'answer': item['answer'],
                'info': {'reference': item['answer']}
            }
            for item in data
        ])

        # initialize combined reward system (hre + pre)
        def correctness_reward(
            prompt: List[Dict],
            completion: List[Dict],
            answer: str,
            info: Dict,
            state: Dict,
            **kwargs
        ) -> float:
            correct_answer = normalize_answer(answer)
            hre = HardRewardEstimator(self.terminal_checker, correct_answer)

            question_text = prompt[0]['content'] if prompt else ""
            response_text = completion[-1]['content'] if completion else ""

            return hre.calculate_reward(question_text, response_text, terminal_only=True)

        def perplexity_reward(
            prompt: List[Dict],
            completion: List[Dict],
            state: Dict,
            **kwargs
        ) -> float:
            if not self.use_student_model or self.student_model is None:
                return 0.0

            pre = PerplexityRewardEstimator(self.student_model, self.terminal_checker)

            question_text = prompt[0]['content'] if prompt else ""
            response_text = completion[-1]['content'] if completion else ""

            return pre.calculate_reward(question_text, response_text, normalize=True)

        # combined weights from config
        hre_weight = config["rewards"]["hre_weight"]
        pre_weight = config["rewards"]["pre_weight"] if use_student_model else 0.0

        rubric = vf.Rubric(
            funcs=[correctness_reward, perplexity_reward],
            weights=[hre_weight, pre_weight]
        )

        # high safety limit for max_turns (actual depth controlled by is_completed)
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=100,
            **kwargs
        )

    async def env_response(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> str:
        # for single-agent mcts evaluation, no environment response needed
        # agent continues reasoning until terminal or depth limit
        return ""

    async def is_completed(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> bool:
        # safety guard from base class
        if await super().is_completed(messages, state, **kwargs):
            return True

        # adaptive depth limit (scales with compute)
        current_depth = state.get('depth', 0)
        if current_depth >= self.max_tree_depth:
            return True

        # problem solved by terminal checker
        if state.get('terminal', False):
            return True

        # mcts tree exhausted (no more promising branches)
        if state.get('mcts_exhausted', False):
            return True

        # optional token budget exhausted
        if self.max_tokens > 0:
            total_tokens = state.get('total_tokens', 0)
            if total_tokens >= self.max_tokens:
                return True

        return False
