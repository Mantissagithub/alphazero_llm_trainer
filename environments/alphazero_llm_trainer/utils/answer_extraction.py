import re
from typing import Optional, Union


def extract_numerical_answer(text: str) -> Optional[float]:
    text = text.strip()

    gsm8k_match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", text)
    if gsm8k_match:
        number_str = gsm8k_match.group(1).replace(",", "")
        return float(number_str)

    patterns = [
        r"(?:the answer is|answer:|final answer:)\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        r"=\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$",
        r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            number_str = match.group(1).replace(",", "")
            return float(number_str)

    all_numbers = re.findall(r"-?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if all_numbers:
        last_number = all_numbers[-1].replace(",", "")
        return float(last_number)

    return None


def normalize_answer(answer: Union[str, float, int]) -> Optional[float]:
    if isinstance(answer, (int, float)):
        return float(answer)

    if isinstance(answer, str):
        return extract_numerical_answer(answer)

    return None


def compare_answers(
    answer1: Union[str, float, int],
    answer2: Union[str, float, int],
    tolerance: float = 1e-6
) -> bool:

    norm1 = normalize_answer(answer1)
    norm2 = normalize_answer(answer2)

    if norm1 is None or norm2 is None:
        return False

    return abs(norm1 - norm2) < tolerance


def extract_step_by_step(text: str) -> list:
    steps = []

    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if re.match(r"^\d+[\.\)]\s", line) or re.match(r"^step\s*\d+", line, re.IGNORECASE):
            steps.append(line)
        elif "=" in line or any(op in line for op in ['+', '-', '*', '/', '×', '÷']):
            steps.append(line)

    return steps


def validate_numerical_format(answer: Union[str, float, int]) -> bool:
    normalized = normalize_answer(answer)
    return normalized is not None
