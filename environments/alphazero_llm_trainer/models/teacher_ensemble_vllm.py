from typing import List, Optional
from vllm import LLM, SamplingParams
import torch


class VLLMTeacherEnsemble:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.teachers = []

        teacher_configs = [
            "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
            "unsloth/gemma-2-9b-it-bnb-4bit"
        ]
        # teacher_configs = [
        #     "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        #     "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        #     "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        #     # "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
        #     "unsloth/gemma-2-9b-it-bnb-4bit"
        # ]


        print("\n" + "=" * 70)
        print("loading vllm teacher ensemble")
        print("=" * 70)

        for idx, model_name in enumerate(teacher_configs, 1):
            short_name = model_name.split('/')[-1]
            print(f"[{idx}/4] initializing {short_name}...")

            llm = LLM(
              model=model_name,
              quantization="bitsandbytes",
              dtype="bfloat16",
              gpu_memory_utilization=0.10,
              max_model_len=1024,
              enforce_eager=False,
              trust_remote_code=True,
              disable_log_stats=False,
              tensor_parallel_size=1,
              max_num_seqs=4,
            )

            self.teachers.append({
                "llm": llm,
                "name": short_name,
                "full_name": model_name
            })

            print(f"    ✓ Ready (vLLM optimized)")

        print("=" * 70)
        print(f"✓ All {len(self.teachers)} teachers loaded")
        print("=" * 70)

    def generate(
        self,
        prompt: str,
        model_index: Optional[int] = None,
        max_tokens: int = 1,
        temperature: float = 0.8,
        **kwargs
    ) -> str:
        import random

        if model_index is None:
            model_index = random.randint(0, len(self.teachers) - 1)

        teacher = self.teachers[model_index]
        llm = teacher["llm"]

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            skip_special_tokens=True
        )

        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        return generated_text

    def generate_batch(
        self,
        prompt: str,
        num_generations: int = 5,
        max_tokens: int = 1,
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
        import random

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
