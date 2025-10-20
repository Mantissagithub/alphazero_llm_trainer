from typing import List, Dict, Optional
from openai import OpenAI
import numpy as np
from config import get_teacher_models_config


class TeacherEnsemble:
    def __init__(
        self,
        client: OpenAI,
        tier: str = "free",
        num_models: Optional[int] = None
    ):
        self.client = client
        self.tier = tier
        self.all_models = get_teacher_models_config(tier)
        self.num_models = num_models or len(self.all_models)
        self.active_models = self._select_models()
        self.model_weights = {m["name"]: 1.0 for m in self.active_models}

    def _select_models(self) -> List[Dict]:
        if self.num_models >= len(self.all_models):
            return self.all_models
        return list(np.random.choice(self.all_models, self.num_models, replace=False))

    def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        if model_name is None:
            model_name = np.random.choice([m["name"] for m in self.active_models])

        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            return ""

    def generate_batch(
        self,
        prompt: str,
        num_generations: int,
        max_tokens: int = 100,
        temperature: float = 0.7,
        diverse: bool = True
    ) -> List[str]:
        outputs = []
        models_to_use = self.active_models if diverse else [self.active_models[0]] * num_generations

        for model_info in models_to_use[:num_generations]:
            output = self.generate(
                prompt=prompt,
                model_name=model_info["name"],
                max_tokens=max_tokens,
                temperature=temperature
            )
            if output:
                outputs.append(output)

        return outputs

    def evaluate(
        self,
        prompt: str,
        response: str,
        num_evaluators: int = 3
    ) -> float:
        scores = []
        evaluators = np.random.choice(self.active_models, min(num_evaluators, len(self.active_models)), replace=False)

        eval_prompt = f"""Rate this response on a scale of 0.0 to 1.0:

Question: {prompt}
Response: {response}

Score (0.0-1.0):"""

        for evaluator in evaluators:
            try:
                score_text = self.generate(
                    prompt=eval_prompt,
                    model_name=evaluator["name"],
                    max_tokens=10,
                    temperature=0.0
                )
                score = float(score_text.strip())
                scores.append(max(0.0, min(1.0, score)))
            except:
                continue

        return np.mean(scores) if scores else 0.5

    def get_model_names(self) -> List[str]:
        return [m["name"] for m in self.active_models]

    def update_model_weight(self, model_name, reward, direction):
        if direction == "right" and reward > 0.5:
            self.model_weights[model_name] *= 1.1
        elif direction == "wrong" and reward > 0.5:
            self.model_weights[model_name] *= 0.9
        elif direction == "right" and reward < 0.5:
            self.model_weights[model_name] *= 0.95
        self.model_weights[model_name] = max(0.1, min(5.0, self.model_weights[model_name]))

    def sample_model_weighted(self):
        names = [m["name"] for m in self.active_models]
        weights = [self.model_weights[n] for n in names]
        probs = [w/sum(weights) for w in weights]
        return np.random.choice(names, p=probs)

