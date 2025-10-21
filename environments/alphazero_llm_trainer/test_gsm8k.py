import argparse
import re
from typing import Tuple
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel


def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3.2-3b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def format_prompt(question: str) -> str:
    return f"Question: {question}\n\nLet's solve this step by step:\n"


def extract_answer(response: str, ground_truth: str) -> Tuple[str, str]:
    pred_matches = re.findall(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)', response)
    pred = pred_matches[-1].replace(',', '') if pred_matches else "0"

    gt_match = re.search(r'####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)', ground_truth)
    if gt_match:
        gt = gt_match.group(1).replace(',', '')
    else:
        gt_nums = re.findall(r'(-?\d+(?:,\d{3})*(?:\.\d+)?)', ground_truth)
        gt = gt_nums[-1].replace(',', '') if gt_nums else "0"

    return pred, gt


def evaluate_example(model, tokenizer, question: str, ground_truth: str, max_new_tokens: int = 256):
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    pred, gt = extract_answer(response, ground_truth)
    is_correct = pred == gt

    return is_correct, response, pred, gt


def main(args):
    model, tokenizer = load_base_model()

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_examples:
        dataset = dataset.select(range(args.num_examples))

    correct = 0
    total = len(dataset)
    results = []

    model.eval()
    print(f"\nTesting BASE Llama-3.2-3B on {total} GSM8K examples\n")

    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = example["answer"]

        is_correct, response, pred, gt = evaluate_example(
            model, tokenizer, question, ground_truth, args.max_new_tokens
        )

        correct += int(is_correct)
        results.append({
            "question": question,
            "predicted_answer": pred,
            "ground_truth": gt,
            "full_response": response.strip(),
            "correct": is_correct
        })

        if (i + 1) % args.log_every == 0:
            accuracy = (correct / (i + 1)) * 100
            print(f"[{i+1}/{total}] Accuracy: {accuracy:.2f}% ({correct}/{i+1})")
            print(f"  Q: {question[:60]}...")
            print(f"  Pred: {pred} | GT: {gt} | {'✓' if is_correct else '✗'}\n")

    final_accuracy = (correct / total) * 100
    print(f"\nFINAL ACCURACY: {final_accuracy:.2f}% ({correct}/{total})\n")

    if args.save_results:
        import json
        with open("gsm8k_base_results.json", "w") as f:
            json.dump({
                "model": "unsloth/llama-3.2-3b-bnb-4bit",
                "total": total,
                "correct": correct,
                "accuracy": final_accuracy,
                "results": results
            }, f, indent=2)
        print("Results saved to gsm8k_base_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--save_results", action="store_true")
    args = parser.parse_args()
    main(args)
