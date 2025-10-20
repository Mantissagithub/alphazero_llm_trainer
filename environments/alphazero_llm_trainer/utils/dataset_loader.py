from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, Dataset
from config import get_training_config


def load_gsm8k_dataset(
    split: str = "train",
    num_samples: Optional[int] = None
) -> List[Dict[str, str]]:

    dataset = load_dataset("openai/gsm8k", "main", split=split)

    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))

    processed_data = []
    for item in dataset:
        processed_data.append({
            "question": item["question"],
            "answer": item["answer"]
        })

    return processed_data


def load_gsm8k_as_hf_dataset(
    split: str = "train",
    num_samples: Optional[int] = None
) -> Dataset:
    data = load_gsm8k_dataset(split, num_samples)

    formatted = [
        {
            'question': item['question'],
            'answer': item['answer'],
            'info': {'reference': item['answer']}
        }
        for item in data
    ]

    return Dataset.from_list(formatted)


def prepare_dataset_for_training(
    train_size: Optional[int] = None,
    eval_size: Optional[int] = None,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:

    config = get_training_config()

    if train_size is None:
        train_size = config["dataset"]["train_size"]
    if eval_size is None:
        eval_size = config["dataset"]["eval_size"]

    train_data = load_gsm8k_dataset("train", train_size)
    test_data = load_gsm8k_dataset("test", eval_size)

    if config["dataset"]["preprocessing"]["shuffle"]:
        import random
        random.seed(seed)
        random.shuffle(train_data)
        random.shuffle(test_data)

    if config["dataset"]["preprocessing"]["remove_duplicates"]:
        train_data = remove_duplicates(train_data)
        test_data = remove_duplicates(test_data)

    return train_data, test_data


def remove_duplicates(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen_questions = set()
    unique_data = []

    for item in data:
        question = item["question"].strip()
        if question not in seen_questions:
            seen_questions.add(question)
            unique_data.append(item)

    return unique_data


def create_prompt_template(question: str, answer: Optional[str] = None) -> str:
    if answer:
        return f"Question: {question}\n\nAnswer: {answer}"
    else:
        return f"Question: {question}\n\nAnswer:"


def format_dataset_for_training(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    formatted = []
    for item in data:
        formatted.append({
            "prompt": create_prompt_template(item["question"]),
            "completion": item["answer"],
            "full_text": create_prompt_template(item["question"], item["answer"])
        })
    return formatted


def get_dataset_statistics(data: List[Dict[str, str]]) -> Dict:
    if not data:
        return {}

    question_lengths = [len(item["question"]) for item in data]
    answer_lengths = [len(item["answer"]) for item in data]

    return {
        "num_samples": len(data),
        "avg_question_length": sum(question_lengths) / len(question_lengths),
        "avg_answer_length": sum(answer_lengths) / len(answer_lengths),
        "max_question_length": max(question_lengths),
        "max_answer_length": max(answer_lengths)
    }
