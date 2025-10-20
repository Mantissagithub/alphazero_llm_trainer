from typing import Optional
from openai import OpenAI
from config import get_model_config
from prompts import check_terminal_state_prompt, extract_answer_prompt, get_baseline_perplexity_prompt


class TerminalChecker:
    def __init__(self, client: OpenAI, model_name: Optional[str] = None):
        self.client = client
        self.config = get_model_config()
        self.model_name = model_name or self.config["terminal_checker"]["name"]

    def is_terminal(self, question: str, response: str) -> bool:
        if not response.strip():
            return False

        prompt = check_terminal_state_prompt(question, response)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0
            )
            answer = result.choices[0].message.content.strip().lower()
            return answer == "yes"
        except:
            return False

    def extract_answer(self, question: str, response: str) -> Optional[str]:
        prompt = extract_answer_prompt(question, response)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0
            )
            answer = result.choices[0].message.content.strip()

            if answer.upper() == "NONE":
                return None

            return answer
        except:
            return None

    def get_baseline_completion(self, question: str, current_state: str, max_tokens: int = 50) -> str:
        # Use the imported function from module level
        prompt = get_baseline_perplexity_prompt(question, current_state)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return result.choices[0].message.content.strip()
        except:
            return ""
