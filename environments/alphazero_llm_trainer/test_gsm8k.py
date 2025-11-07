import argparse
import re
from typing import Tuple
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel
import json

def load_base_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3.2-3b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def load_trained_model():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./student_final_model2",
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

def evaluate_model(model, tokenizer, dataset, model_name="model", num_examples=None):
    if num_examples:
        test_data = dataset[:num_examples]
    else:
        test_data = dataset

    correct = 0
    results = []

    for idx, example in enumerate(test_data):
        question = example["question"]
        ground_truth = example["answer"]

        prompt = format_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred, gt = extract_answer(response, ground_truth)

        is_correct = pred == gt
        if is_correct:
            correct += 1

        results.append({
            "question": question,
            "predicted": pred,
            "ground_truth": gt,
            "correct": is_correct
        })

        # Print progress
        if (idx + 1) % 10 == 0:
            accuracy = correct / (idx + 1) * 100
            print(f"[{model_name}] Progress: {idx+1}/{len(test_data)} | Accuracy: {accuracy:.2f}%")

    accuracy = correct / len(test_data) * 100
    print(f"\n[{model_name}] Final Accuracy: {accuracy:.2f}% ({correct}/{len(test_data)})")

    return accuracy, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=None, help="Number of examples to test")
    args = parser.parse_args()

    # Load dataset
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    # Evaluate base model
    print("\n" + "="*60)
    print("EVALUATING BASE MODEL: unsloth/llama-3.2-3b-bnb-4bit")
    print("="*60)
    model_base, tokenizer_base = load_base_model()
    base_accuracy, base_results = evaluate_model(
        model_base, tokenizer_base, test_data,
        model_name="BASE",
        num_examples=args.num_examples
    )
    del model_base
    # base_accuracy = 13.58
    torch.cuda.empty_cache()

    # Evaluate trained model
    print("\n" + "="*60)
    print("EVALUATING TRAINED MODEL: ./student_final_model (AlphaZero LoRA)")
    print("="*60)
    model_trained, tokenizer_trained = load_trained_model()
    trained_accuracy, trained_results = evaluate_model(
        model_trained, tokenizer_trained, test_data,
        model_name="TRAINED",
        num_examples=args.num_examples
    )
    del model_trained
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Base Model Accuracy:    {base_accuracy:.2f}%")
    print(f"Trained Model Accuracy: {trained_accuracy:.2f}%")
    print(f"Improvement:            {trained_accuracy - base_accuracy:+.2f}%")
    print("="*60)

    # Save results
    comparison_results = {
        "base_accuracy": base_accuracy,
        "trained_accuracy": trained_accuracy,
        "improvement": trained_accuracy - base_accuracy,
        "num_examples": args.num_examples or len(test_data),
        "base_results": base_results,
        "trained_results": trained_results
    }

    with open("gsm8k_comparison_results.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    print("Results saved to gsm8k_comparison_results.json")

if __name__ == "__main__":
    main()
