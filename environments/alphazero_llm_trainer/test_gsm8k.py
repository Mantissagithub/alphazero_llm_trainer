import argparse
import re
from typing import List, Tuple
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from peft import PeftModel
import yaml

def load_config():
    config = {
        "model_name": "unsloth/llama-3.2-3b-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "dtype": None,
        "lora_config": {
            "r": 32,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "use_gradient_checkpointing": True,
            "random_state": 42,
        }
    }
    return config

def load_model(checkpoint_path: str, config: dict) -> Tuple[FastLanguageModel, any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model_name"],
        max_seq_length=config["max_seq_length"],
        dtype=config["dtype"],
        load_in_4bit=config["load_in_4bit"],
    )

    if checkpoint_path and checkpoint_path != "none":
        # Load PEFT/LoRA adapter
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora_config"]["r"],
            target_modules=config["lora_config"]["target_modules"],
            lora_alpha=config["lora_config"]["lora_alpha"],
            lora_dropout=config["lora_config"]["lora_dropout"],
            bias=config["lora_config"]["bias"],
            use_gradient_checkpointing=config["lora_config"]["use_gradient_checkpointing"],
            random_state=config["lora_config"]["random_state"],
        )
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print(f"Loaded fine-tuned LoRA from {checkpoint_path}")
    else:
        print("No checkpoint provided; using base model for baseline.")

    FastLanguageModel.for_inference(model)
    return model, tokenizer

def format_prompt(question: str) -> str:
    return f"Question: {question}\nLet's think step by step.\nAnswer:"

def extract_answer(response: str, ground_truth: str) -> Tuple[str, str]:
    # Extract predicted answer: last number or boxed content
    pred_match = re.search(r'(-?\d+(?:\.\d+)?)', response[::-1])
    pred = pred_match.group(1)[::-1] if pred_match else "0"

    gt = re.sub(r'[^0-9.-]', '', ground_truth)

    return pred, gt

def evaluate_example(model, tokenizer, question: str, ground_truth: str, max_new_tokens: int = 256) -> bool:
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
    print(f"Q: {question[:50]}...\nPred: {response.strip()}\nExtracted: {pred} (GT: {gt}) -> {'Correct' if is_correct else 'Wrong'}\n")
    return is_correct, response

def main(args):
    config = load_config()
    model, tokenizer = load_model(args.checkpoint, config)

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    if args.num_examples:
        dataset = dataset.select(range(args.num_examples))

    correct = 0
    total = len(dataset)
    responses = []

    model.eval()
    for i, example in enumerate(dataset):
        question = example["question"]
        ground_truth = example["answer"]

        is_correct, response = evaluate_example(
            model, tokenizer, question, ground_truth, args.max_new_tokens
        )
        correct += int(is_correct)
        responses.append({"question": question, "pred": response, "gt": ground_truth, "correct": is_correct})

        if (i + 1) % args.batch_size == 0:
            accuracy = (correct / (i + 1)) * 100
            print(f"Progress: {i+1}/{total} | Accuracy so far: {accuracy:.2f}%")

    final_accuracy = (correct / total) * 100
    print(f"\nFinal Results on GSM8K Test ({total} examples):")
    print(f"Accuracy: {final_accuracy:.2f}% ({correct}/{total} correct)")

    import json
    with open("gsm8k_eval_results.json", "w") as f:
        json.dump(responses, f, indent=2)
    print("Detailed results saved to gsm8k_eval_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LoRA model on GSM8K")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/student_final", help="Path to PEFT checkpoint (or 'none' for base)")
    parser.add_argument("--num_examples", type=int, default=None, help="Limit to N examples (default: full test set)")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens for generation")
    parser.add_argument("--batch_size", type=int, default=1, help="Log progress every N examples")
    args = parser.parse_args()
    main(args)
