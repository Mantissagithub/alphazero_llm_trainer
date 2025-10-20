import torch
import numpy as np
from typing import Optional

try:
    from models import StudentModel, TerminalChecker
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models import StudentModel, TerminalChecker


class PerplexityRewardEstimator:
    def __init__(
        self,
        student_model: StudentModel,
        terminal_checker: TerminalChecker,
        baseline_perplexity: Optional[float] = None
    ):
        self.student_model = student_model
        self.terminal_checker = terminal_checker
        self.baseline_perplexity = baseline_perplexity
        self._perplexity_cache = {}

    def calculate_perplexity(self, text: str) -> float:
        if text in self._perplexity_cache:
            return self._perplexity_cache[text]

        inputs = self.student_model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.student_model.max_seq_length
        ).to(self.student_model.model.device)

        with torch.no_grad():
            outputs = self.student_model.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        self._perplexity_cache[text] = perplexity
        return perplexity

    def set_baseline(self, question: str, reference_answer: str):
        baseline_text = f"{question}\n{reference_answer}"
        self.baseline_perplexity = self.calculate_perplexity(baseline_text)

    def calculate_reward(self, question: str, response: str, normalize: bool = True) -> float:
        if not response.strip():
            return -1.0

        full_text = f"{question}\n{response}"
        current_perplexity = self.calculate_perplexity(full_text)

        if self.baseline_perplexity is None:
            baseline_completion = self.terminal_checker.get_baseline_completion(question, "")
            self.set_baseline(question, baseline_completion)

        if normalize:
            reward = (self.baseline_perplexity - current_perplexity) / (self.baseline_perplexity + 1e-8)
            return np.clip(reward, -1.0, 1.0)
        else:
            return -current_perplexity

    def get_token_level_perplexity(self, text: str) -> list:
        inputs = self.student_model.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.student_model.max_seq_length
        ).to(self.student_model.model.device)

        with torch.no_grad():
            outputs = self.student_model.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        token_perplexities = torch.exp(token_losses).cpu().numpy().tolist()
        return token_perplexities

    def clear_cache(self):
        self._perplexity_cache.clear()
