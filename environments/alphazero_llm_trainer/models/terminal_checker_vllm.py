from typing import Optional
from vllm import LLM, SamplingParams
from prompts import check_terminal_state_prompt, extract_answer_prompt, get_baseline_perplexity_prompt


class VLLMTerminalChecker:
    def __init__(self, device="cuda:0"):
        self.device = device
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        print("\nloading terminal checker (vllm)...")

        self.llm = LLM(
            model=model_name,
            dtype="float16",
            gpu_memory_utilization=0.05,
            max_model_len=512,
            enforce_eager=True,
            trust_remote_code=True,
            disable_log_stats=True
        )

        print("✓ terminal checker ready")

    def _generate(self, prompt: str, max_tokens: int = 10) -> str:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_tokens,
            skip_special_tokens=True
        )

        outputs = self.llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text.strip()

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
