import os
from typing import Dict, Optional, Any, List
from openai import OpenAI
import verifiers as vf
from datasets import Dataset

from models import TeacherEnsemble, StudentModel, TerminalChecker
from rewards import HardRewardEstimator, PerplexityRewardEstimator
from utils import load_gsm8k_dataset, normalize_answer
from config import get_training_config


class AlphaZeroLLMEnvironment(vf.SingleTurnEnv):
    def __init__(
        self,
        tier: str = "free",
        use_student_model: bool = False,
        **kwargs
    ):
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

        self.tier = tier
        self.use_student_model = use_student_model

        self.terminal_checker = TerminalChecker(self.client)
        self.student_model = None
        if use_student_model:
            self.student_model = StudentModel()

        config = get_training_config()
        train_size = config["dataset"]["train_size"]
        data = load_gsm8k_dataset("train", train_size)

        dataset = Dataset.from_list([
            {
                'question': item['question'],
                'answer': item['answer'],
                'info': {'reference': item['answer']}
            }
            for item in data
        ])

        def correctness_reward(prompt: List[Dict], completion: List[Dict], answer: str, info: Dict, **kwargs) -> float:
            correct_answer = normalize_answer(answer)
            hre = HardRewardEstimator(self.terminal_checker, correct_answer)

            question_text = prompt[0]['content']
            response_text = completion[-1]['content'] if completion else ""

            return hre.calculate_reward(question_text, response_text, terminal_only=True)

        def perplexity_reward(prompt: List[Dict], completion: List[Dict], state: Dict, **kwargs) -> float:
            if not self.use_student_model or self.student_model is None:
                return 0.0

            pre = PerplexityRewardEstimator(self.student_model, self.terminal_checker)

            question_text = prompt[0]['content']
            response_text = completion[-1]['content'] if completion else ""

            return pre.calculate_reward(question_text, response_text, normalize=True)

        weights = [1.0, 0.6] if use_student_model else [1.0, 0.0]

        rubric = vf.Rubric(
            funcs=[correctness_reward, perplexity_reward],
            weights=weights
        )

        super().__init__(dataset=dataset, rubric=rubric, **kwargs)


def load_environment(**kwargs) -> vf.Environment:
    return AlphaZeroLLMEnvironment(**kwargs)
