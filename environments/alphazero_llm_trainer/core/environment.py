import os
import asyncio
import re
import logging
import math
from typing import Dict, Optional, Any, List, Tuple, Callable
import verifiers as vf
from verifiers.types import Messages, State
from datasets import Dataset
from openai import AsyncOpenAI
# from verifiers.parser import BaseParser

logger = logging.getLogger(__name__)

from models.student_model import StudentModel
from models.terminal_checker_vllm import VLLMTerminalChecker
from rewards.hre import HardRewardEstimator
from rewards.pre import PerplexityRewardEstimator
from rewards.combined import CombinedRewardSystem
from utils.dataset_loader import load_gsm8k_dataset
from utils import normalize_answer
from config import get_training_config

GSM8K_SYSTEM_PROMPT = """You are a mathematical problem solver.
Read each problem carefully and solve it step by step.
Provide your reasoning and final numerical answer.
Format your final answer as: #### [number]"""

def gsm8k_feedback_fn(observation: str) -> str:
    return observation.strip()

class GSM8M_Parser(vf.Parser):
    def parse_answer(self, completion: Messages) -> Optional[str]:
        response = ""
        for msg in reversed(completion):
            if msg['role'] == 'assistant' and msg.get('content'):
                response = msg['content'].strip()
                break
        if not response:
            return None

        match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', response)
        if match:
            return normalize_answer(match.group(1).replace(',', ''))

        # Try LaTeX boxed format $\boxed{...}$
        match = re.search(r'\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}', response)
        if match:
            return normalize_answer(match.group(1).replace(',', ''))

        # Try "Final Answer: ..." or "Answer: ..."
        match = re.search(r'(?:final\s+)?answer[:\s]+(-?\d+(?:,\d{3})*(?:\.\d+)?)', response, re.IGNORECASE)
        if match:
            return normalize_answer(match.group(1).replace(',', ''))

        numbers = re.findall(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)', response)
        return normalize_answer(numbers[-1].replace(',', '')) if numbers else None

    def get_format_reward_func(self) -> Callable:
        async def format_reward(
            prompt: Messages,
            completion: Messages,
            answer: str,
            state: State,
            **kwargs
        ) -> float:
            resp = ""
            for msg in reversed(completion):
                if msg['role'] == 'assistant' and msg.get('content'):
                    resp = msg['content'].strip()
                    break

            if not resp:
                return 0.0

            if "####" in resp or r"\boxed{" in resp or r"$\boxed{" in resp:
                return 1.0
            if any(re.search(r'\d+', msg.get("content", "")) for msg in completion if msg["role"] == "assistant" and msg.get("content")):
                return 0.5
            return 0.0
        return format_reward


class AlphaZeroLLMEnvironment(vf.MultiTurnEnv):
    def __init__(
        self,
        tier: str = "free",
        use_student_model: bool = False,
        num_train_examples: int = 2000,
        num_eval_examples: int = 100,
        use_combined: bool = False,
        hre_weight: float = 1.0,
        pre_weight: float = 0.6,
        format_weight: float = 0.2,
        efficiency_weight: float = 0.1,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        self.tier = tier
        self.use_student_model = use_student_model

        # load config for defaults
        config = get_training_config()

        # adaptive depth from env var (overrides config for compute scaling)
        self.max_tree_depth = int(os.environ.get(
            'MAX_TREE_DEPTH',
            config.get('mcts', {}).get('max_tree_depth', 10)
        ))

        # optional token budget from env var
        self.max_tokens = int(os.environ.get('MAX_TOKENS', 0))

        # mcts iterations configurable via env var
        self.num_mcts_iterations = int(os.environ.get(
            'NUM_MCTS_ITERATIONS',
            config.get('mcts', {}).get('num_iterations', 100)
        ))

        # initialize vllm terminal checker
        self.terminal_checker = VLLMTerminalChecker()

        # student model for perplexity rewards
        self.student_model = None
        if use_student_model:
            self.student_model = StudentModel()

        # load dataset with verifiers schema (prompt/answer)
        train_size = num_train_examples or config.get("dataset", {}).get("train_size", 2000)
        train_data = load_gsm8k_dataset("train", train_size)
        eval_data = load_gsm8k_dataset("test", num_eval_examples)

        self.train_dataset = Dataset.from_list([
            {
                'prompt': [{'role': 'user', 'content': item['question']}],
                'answer': item['answer'],
                'info': {'reference': item['answer'], 'full_reference': item.get('full_answer', '')}
            }
            for item in train_data
        ])

        self.eval_dataset = Dataset.from_list([
            {
                'prompt': [{'role': 'user', 'content': item['question']}],
                'answer': item['answer'],
                'info': {'reference': item['answer'], 'full_reference': item.get('full_answer', '')}
            }
            for item in eval_data
        ])

        # initialize combined reward system (hre + pre)
        async def correctness_reward(
            prompt: Messages,
            completion: Messages,
            answer: str,
            state: State,
            **kwargs
        ) -> float:
            correct_answer = normalize_answer(answer)
            hre = HardRewardEstimator(self.terminal_checker, correct_answer)

            question_text = prompt[0]['content'] if prompt else ""
            response_text = completion[-1]['content'] if completion else ""

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: hre.calculate_reward(question_text, response_text, terminal_only=True)
            )

        # async def baseline_perplexity_reward(
        #     prompt: Messages,
        #     completion: Messages,
        #     answer: str,
        #     state: State,
        #     **kwargs
        # ) -> float:
        #     question_text = prompt[0]['content'] if prompt else ""
        #     response_text = completion[-1]['content'] if completion else ""

        #     if not response_text:
        #         return 0.0

        #     try:
        #         api_key = os.environ.get("PRIME_API_KEY", "")
        #         if not api_key:
        #             return 0.0

        #         base_url = os.environ.get("PRIME_INFERENCE_URL", "https://api.pinference.ai/api/v1")
        #         client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        #         response_obj = await client.chat.completions.create(
        #             model="z-ai/glm-4.5-air",
        #             messages=[
        #                 {"role": "user", "content": question_text},
        #                 {"role": "assistant", "content": response_text[:20]},
        #             ],
        #             max_tokens=max(1, len(response_text) // 2),
        #             temperature=0.2,
        #             logprobs=True,
        #             top_logprobs=1
        #         )

        #         print(f"response obj: {response_obj}")

        #         if hasattr(response_obj.choices[0], 'logprobs') and response_obj.choices[0].logprobs:
        #             logprobs_content = response_obj.choices[0].logprobs.content
        #             if logprobs_content and len(logprobs_content) > 0:
        #                 total_logprob = sum(lp.logprob for lp in logprobs_content)
        #                 avg_logprob = total_logprob / len(logprobs_content)
        #                 perplexity = math.exp(-avg_logprob)
        #                 state['baseline_perplexity'] = perplexity
        #                 return perplexity
        #     except Exception as e:
        #         logger.debug(f"baseline perplexity calculation failed: {e}")


        async def perplexity_reward(
            prompt: Messages,
            completion: Messages,
            answer: str,
            state: State,
            **kwargs
        ) -> float:
            if not self.use_student_model or self.student_model is None:
                return 0.0

            pre = PerplexityRewardEstimator(self.student_model, self.terminal_checker)

            question_text = prompt[0]['content'] if prompt else ""
            response_text = completion[-1]['content'] if completion else ""

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: pre.calculate_reward(question_text, response_text, normalize=True)
            )

        async def combined_reward(
            prompt: Messages,
            completion: Messages,
            answer: str,
            state: State,
            **kwargs
        ) -> float:
            if not use_combined or not self.use_student_model:
                return 0.0

            correct_answer = normalize_answer(answer)
            hre = HardRewardEstimator(self.terminal_checker, correct_answer)
            pre = PerplexityRewardEstimator(self.student_model, self.terminal_checker)

            combined = CombinedRewardSystem(
                hre=hre,
                pre=pre,
                hre_weight=hre_weight,
                pre_weight=pre_weight
            )

            question_text = prompt[0]['content'] if prompt else ""
            response_text = completion[-1]['content'] if completion else ""
            path_length = state.get('path_length', None)
            binary_code = state.get('binary_code', None)

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: combined.calculate_reward(
                    question=question_text,
                    response=response_text,
                    path_length=path_length,
                    binary_code=binary_code
                )
            )

            return result["total"]

        async def efficiency_reward(
            prompt: Messages,
            completion: Messages,
            answer: str,
            state: State,
            **kwargs
        ) -> float:
            if efficiency_weight <= 0:
                return 0.0

            num_turns = len([msg for msg in completion if msg["role"] == "assistant"])
            hre_score = await correctness_reward(prompt, completion, answer, state, **kwargs)

            if hre_score > 0.5:
                return max(0.0, 1.0 - (num_turns * 0.1))
            return 0.0

        reward_funcs: List[Callable] = []
        reward_weights: List[float] = []
        if use_combined and self.use_student_model:
            reward_funcs.append(combined_reward)
            reward_weights.append(1.0)
        else:
            reward_funcs.append(correctness_reward)
            reward_weights.append(hre_weight or config.get("rewards", {}).get("hre_weight", 1.0))

            if use_student_model:
                reward_funcs.append(perplexity_reward)
                reward_weights.append(pre_weight or config.get("rewards", {}).get("pre_weight", 0.6))
            else:
                # add baseline perplexity tracking when not using student model
                # reward_funcs.append(baseline_perplexity_reward)
                reward_weights.append(0.0)

        parser = GSM8M_Parser()
        reward_funcs.append(parser.get_format_reward_func())
        reward_weights.append(format_weight or 0.2)

        if efficiency_weight > 0:
            reward_funcs.append(efficiency_reward)
            reward_weights.append(efficiency_weight)

        # Log parser information before creating Rubric
        print("=" * 80)
        print("Parser Information:")
        print(f"  Parser type: {type(parser).__name__}")
        print(f"  Parser class: {type(parser)}")
        print(f"  Parser instance id: {id(parser)}")
        print(f"  Parser module: {type(parser).__module__}")
        if hasattr(parser, '__dict__'):
            print(f"  Parser attributes: {list(parser.__dict__.keys())}")
        print("=" * 80)
        logger.info("=" * 80)
        logger.info("Parser Information:")
        logger.info(f"  Parser type: {type(parser).__name__}")
        logger.info(f"  Parser class: {type(parser)}")
        logger.info(f"  Parser instance id: {id(parser)}")
        logger.info(f"  Parser module: {type(parser).__module__}")
        if hasattr(parser, '__dict__'):
            logger.info(f"  Parser attributes: {list(parser.__dict__.keys())}")
        logger.info("=" * 80)

        self.rubric = vf.Rubric(
            parser = parser,
            funcs = reward_funcs,
            weights = reward_weights
        )

        # Also set parser on environment to match rubric parser
        # This prevents warnings from the verifiers framework
        self.parser = parser

        # Log rubric parser information after creating Rubric
        if hasattr(self.rubric, 'parser'):
            rubric_parser = self.rubric.parser
            print("=" * 80)
            print("Rubric Parser Information:")
            print(f"  Rubric parser type: {type(rubric_parser).__name__}")
            print(f"  Rubric parser class: {type(rubric_parser)}")
            print(f"  Rubric parser instance id: {id(rubric_parser)}")
            print(f"  Rubric parser module: {type(rubric_parser).__module__}")
            if hasattr(rubric_parser, '__dict__'):
                print(f"  Rubric parser attributes: {list(rubric_parser.__dict__.keys())}")

            # Compare parsers
            print("=" * 80)
            print("Parser Comparison:")
            print(f"  Parser instance id: {id(parser)}")
            print(f"  Rubric parser instance id: {id(rubric_parser)}")
            print(f"  Are same instance? {parser is rubric_parser}")
            print(f"  Are same type? {type(parser) is type(rubric_parser)}")
            print(f"  Are equal? {parser == rubric_parser if hasattr(parser, '__eq__') else 'N/A'}")

            if parser is not rubric_parser:
                print("⚠️  WARNING: Parser and rubric parser are DIFFERENT instances!")
                print(f"    This may cause unexpected behavior.")
            print("=" * 80)

            logger.info("=" * 80)
            logger.info("Rubric Parser Information:")
            logger.info(f"  Rubric parser type: {type(rubric_parser).__name__}")
            logger.info(f"  Rubric parser class: {type(rubric_parser)}")
            logger.info(f"  Rubric parser instance id: {id(rubric_parser)}")
            logger.info(f"  Rubric parser module: {type(rubric_parser).__module__}")
            if hasattr(rubric_parser, '__dict__'):
                logger.info(f"  Rubric parser attributes: {list(rubric_parser.__dict__.keys())}")

            # Compare parsers
            logger.info("=" * 80)
            logger.info("Parser Comparison:")
            logger.info(f"  Parser instance id: {id(parser)}")
            logger.info(f"  Rubric parser instance id: {id(rubric_parser)}")
            logger.info(f"  Are same instance? {parser is rubric_parser}")
            logger.info(f"  Are same type? {type(parser) is type(rubric_parser)}")
            logger.info(f"  Are equal? {parser == rubric_parser if hasattr(parser, '__eq__') else 'N/A'}")

            if parser is not rubric_parser:
                logger.warning("⚠️  WARNING: Parser and rubric parser are DIFFERENT instances!")
                logger.warning(f"    This may cause unexpected behavior.")
            logger.info("=" * 80)
        else:
            print("⚠️  WARNING: Rubric does not have a 'parser' attribute")
            print(f"  Rubric attributes: {dir(self.rubric)}")
            logger.warning("Rubric does not have a 'parser' attribute")
            logger.info(f"  Rubric attributes: {dir(self.rubric)}")

        system_prompt = system_prompt or GSM8K_SYSTEM_PROMPT

        # Use eval_dataset if available, otherwise fallback to train_dataset
        eval_dataset = kwargs.pop('eval_dataset', None) or self.eval_dataset

        super().__init__(
            dataset=self.train_dataset,
            eval_dataset=eval_dataset,
            system_prompt=system_prompt,
            rubric=self.rubric,
            feedback_fn=gsm8k_feedback_fn,
            max_turns=100,
            **kwargs
        )

    async def env_response(
        self,
        messages: Messages,
        state: State,
        **kwargs
    ) -> Tuple[Messages, State]:
        # for single-agent mcts evaluation, no environment response needed
        # agent continues reasoning until terminal or depth limit
        return [], state

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

        # check for terminal state by examining last assistant message
        question_text = ""
        last_response = ""

        for msg in messages:
            if msg['role'] == 'user' and not question_text:
                question_text = msg.get('content', '')

        for msg in reversed(messages):
            if msg['role'] == 'assistant' and msg.get('content') and msg['content'].strip():
                last_response = msg.get('content', '').strip()
                break

        # stop early if multiple empty messages (agent is stuck)
        empty_count = sum(1 for msg in messages if msg.get('role') == 'assistant' and (not msg.get('content') or not msg.get('content').strip()))
        if empty_count >= 3:
            return True

        if question_text and last_response:
            try:
                is_terminal = self.terminal_checker.is_terminal(question_text, last_response)
                if is_terminal:
                    state['terminal'] = True
                    return True
            except:
                pass

        # problem solved by terminal checker (if already set)
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
