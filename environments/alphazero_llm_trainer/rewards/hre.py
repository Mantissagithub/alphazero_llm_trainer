import re
from typing import Optional, Union
from models import TerminalChecker

class HardRewardEstimator:
    def __init__(self, terminal_checker: TerminalChecker, correct_answer: Union[float, str]):
        self.terminal_checker = terminal_checker
        self.correct_answer = self._normalize_answer(correct_answer)

    def _normalize_answer(self, answer: Union[float, str]) -> float:
        if isinstance(answer, (int, float)):
            return float(answer)

        match = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", str(answer))
        if match:
            return float(match[-1])

        match = re.findall(r"(-?\d+(?:\.\d+)?)", str(answer))
        if match:
            return float(match[-1])

        return 0.0

    def extract_answer(self, response: str) -> Optional[float]:
        match = re.findall(r"####\s*(-?\d+(?:\.\d+)?)", response)
        if match:
            return float(match[-1])

        match = re.findall(r"(-?\d+(?:\.\d+)?)", response)
        if match:
            return float(match[-1])

        return None

    def calculate_reward(self, question: str, response: str, terminal_only: bool = True) -> float:
        if terminal_only and not self.terminal_checker.is_terminal(question, response):
            return 0.0

        extracted = self.extract_answer(response)

        if extracted is None:
            return 0.0

        tolerance = 1e-6
        if abs(extracted - self.correct_answer) < tolerance:
            return 1.0

        return 0.0

    def get_partial_credit(self, question: str, response: str, method: str = "distance") -> float:
        extracted = self.extract_answer(response)

        if extracted is None:
            return 0.0

        if method == "distance":
            distance = abs(extracted - self.correct_answer)
            max_distance = abs(self.correct_answer) + 1.0
            return max(0.0, 1.0 - (distance / max_distance))

        elif method == "log_distance":
            import math
            distance = abs(extracted - self.correct_answer)
            return max(0.0, 1.0 - math.log1p(distance))

        return 0.0
