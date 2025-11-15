from typing import Optional
import os
from openai import OpenAI
from prompts import check_terminal_state_prompt, extract_answer_prompt, get_baseline_perplexity_prompt


class VLLMTerminalChecker:
    def __init__(self, device="cuda:0"):
        self.device = device
        model_name = os.environ.get("TERMINAL_CHECKER_MODEL", "qwen/qwen-2.5-72b-instruct")
        base_url = os.environ.get("PRIME_INFERENCE_URL", "https://api.pinference.ai/api/v1")
        api_key = os.environ.get("PRIME_API_KEY", "")

        print("\nloading terminal checker (Prime Intellect API)...")

        self.model_name = model_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        print("✓ terminal checker ready")

    def _generate(self, prompt: str, max_tokens: int = 10) -> str:
        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0
            )
            return result.choices[0].message.content.strip()
        except:
            return ""

    def is_terminal(self, question: str, response: str) -> bool:
        if not response.strip():
            return False

        prompt = check_terminal_state_prompt(question, response)

        try:
            answer = self._generate(prompt, max_tokens=10).lower()
            return "yes" in answer
        except:
            return False

    def extract_answer(self, question: str, response: str) -> Optional[str]:
        prompt = extract_answer_prompt(question, response)

        try:
            answer = self._generate(prompt, max_tokens=20)
            if answer.upper() == "NONE":
                return None
            return answer
        except:
            return None

    def get_baseline_completion(self, question: str, current_state: str, max_tokens: int = 50) -> str:
        prompt = get_baseline_perplexity_prompt(question, current_state)

        try:
            result = self._generate(prompt, max_tokens=max_tokens)
            return result
        except:
            return ""
