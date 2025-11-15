from typing import List, Optional
import os
import random
from openai import OpenAI


class PrimeTeacherEnsemble:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cuda:0"
    ):
        self.device = device

        if base_url is None:
            base_url = os.environ.get("PRIME_INFERENCE_URL", "https://api.pinference.ai/api/v1")

        if api_key is None:
            api_key = os.environ.get("PRIME_API_KEY", "")

        self.base_url = base_url
        self.api_key = api_key

        # Updated to use models available in Prime Inference API
        # Selected models with good cost/performance balance for reasoning tasks
        self.teacher_configs = [
            "qwen/qwen-2.5-72b-instruct",              # $0.38/$0.4 - Great for math
            "meta-llama/llama-3.1-70b-instruct",       # $0.9/$0.9 - Strong reasoning
            "mistralai/mistral-small-24b-instruct-2501", # $0.8/$0.8 - Good balance
            "deepseek/deepseek-chat",                  # $0.5/$1.5 - Cost-effective reasoning
            "x-ai/grok-4-fast"              # $0.15/$0.6 - Very cheap and fast
        ]

        self.teachers = []

        print("\n" + "=" * 70)
        print("loading Prime-RL teacher ensemble")
        print("=" * 70)

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key if api_key else None
        )

        for idx, model_name in enumerate(self.teacher_configs, 1):
            short_name = model_name.split('/')[-1]
            self.teachers.append({
                "name": short_name,
                "full_name": model_name
            })
            print(f"[{idx}/{len(self.teacher_configs)}] {short_name} (via Prime-RL)")

        print("=" * 70)
        print(f"✓ All {len(self.teachers)} teachers ready")
        print("=" * 70)

    def generate(
        self,
        prompt: str,
        model_index: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
        **kwargs
    ) -> str:
        if model_index is None:
            model_index = random.randint(0, len(self.teachers) - 1)

        teacher = self.teachers[model_index]
        model_name = teacher["full_name"]

        try:
            result = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95
            )
            return result.choices[0].message.content.strip()
        except Exception as e:
            print(f"Warning: Teacher generation failed for {teacher['name']}: {e}")
            return ""

    def generate_batch(
        self,
        prompt: str,
        num_generations: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.8
    ) -> List[str]:
        outputs = []

        for i in range(min(num_generations, len(self.teachers))):
            output = self.generate(
                prompt=prompt,
                model_index=i,
                max_tokens=max_tokens,
                temperature=temperature
            )
            if output:
                outputs.append(output)

        return outputs

    def get_model_names(self) -> List[str]:
        return [t["name"] for t in self.teachers]

    def evaluate(
        self,
        prompt: str,
        response: str,
        num_evaluators: int = 3
    ) -> float:
        scores = []
        evaluator_indices = random.sample(range(len(self.teachers)), min(num_evaluators, len(self.teachers)))

        eval_prompt = f"""Rate this response on a scale of 0.0 to 1.0:

Question: {prompt}
Response: {response}

Score (0.0-1.0):"""

        for idx in evaluator_indices:
            try:
                score_text = self.generate(
                    prompt=eval_prompt,
                    model_index=idx,
                    max_tokens=10,
                    temperature=0.0
                )
                score = float(score_text.strip())
                scores.append(max(0.0, min(1.0, score)))
            except:
                continue

        return sum(scores) / len(scores) if scores else 0.5

