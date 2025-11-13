from typing import Optional
import torch
from torch.optim import AdamW
# unsloth import deferred to runtime (needs gpu access)
from config import get_model_config
import copy


class StudentModel:
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        load_in_4bit: bool = True,
        device: str = "auto",
        learning_rate: Optional[float] = None
    ):
        self.config = get_model_config()
        self.model_name = model_name or self.config["student_model"]["name"]
        self.max_seq_length = max_seq_length or self.config["student_model"]["max_seq_length"]
        self.load_in_4bit = load_in_4bit
        self.device = device

        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        else:
            self.learning_rate = 2e-5

        self.model = None
        self.tokenizer = None
        self._load_model()
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.ref_model = None
        self._create_ref_model()

    def _create_ref_model(self):
        print("creating ref model for kl penalty")

        self.ref_model = copy.deepcopy(self.model)

        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.ref_model.eval()

        # stays on gpu, unsloth needs gpu (triton kernels won't work on cpu)

        print("ref model created")

    def update_ref_model(self):
        print("updating ref model for kl penalty")

        del self.ref_model
        torch.cuda.empty_cache()

        self._create_ref_model()

    def _load_model(self):
        # import unsloth here so gpu check happens at runtime, not build time
        from unsloth import FastLanguageModel

        lora_config = self.config["student_model"]["lora_config"]

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config["r"],
            target_modules=lora_config["target_modules"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
            random_state=lora_config["random_state"],
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        return response

    def get_logits(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        # import unsloth here so gpu check happens at runtime, not build time
        from unsloth import FastLanguageModel

        self.model = FastLanguageModel.from_pretrained(path)
        self.tokenizer = FastLanguageModel.from_pretrained(path)

    def prepare_for_training(self):
        self.model.train()

    def prepare_for_inference(self):
        self.model.eval()

    def compute_loss(self, text, reward):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs, labels=inputs["input_ids"])
        return outputs.loss * reward

    def compute_grpo_loss(
        self,
        trajectories: list,
        beta: float = 0.1,
        clip_advantages: float = 3.0,
        kl_weight: float = 0.01,
        micro_batch_size: int = 2  # small batches to avoid oom
    ) -> torch.Tensor:
        if not trajectories:
            return torch.tensor(0.0, device=self.model.device)

        rewards = torch.tensor(
            [t['reward'] for t in trajectories],
            dtype=torch.float32,
            device=self.model.device
        )

        mean_reward = rewards.mean()
        std_reward = rewards.std() + 1e-8
        advantages = (rewards - mean_reward) / std_reward

        if clip_advantages > 0:
            advantages = torch.clamp(advantages, -clip_advantages, clip_advantages)

        total_loss = 0.0
        total_kl_loss = 0.0

        num_batches = (len(trajectories) + micro_batch_size - 1) // micro_batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * micro_batch_size
            end_idx = min(start_idx + micro_batch_size, len(trajectories))
            batch_trajectories = trajectories[start_idx:end_idx]
            batch_advantages = advantages[start_idx:end_idx]

            for traj, advantage in zip(batch_trajectories, batch_advantages):
                text = traj['text']

                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                    padding=False
                ).to(self.model.device)

                outputs = self.model(**inputs)
                logits = outputs.logits
                del outputs  # free asap

                token_ids = inputs['input_ids'][:, 1:]
                shifted_logits = logits[:, :-1, :]

                log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
                selected_log_probs = log_probs.gather(
                    dim=-1,
                    index=token_ids.unsqueeze(-1)
                ).squeeze(-1)

                pg_loss = -(selected_log_probs.mean() * advantage)

                # ref model on gpu (unsloth needs gpu)
                with torch.no_grad():
                    ref_outputs = self.ref_model(**inputs)
                    ref_logits_shifted = ref_outputs.logits[:, :-1, :].detach()
                    del ref_outputs
                    torch.cuda.empty_cache()  # clear cache aggressively

                ref_log_probs = torch.nn.functional.log_softmax(ref_logits_shifted, dim=-1)

                # kl div in log space (saves memory)
                kl_div = torch.nn.functional.kl_div(
                    ref_log_probs,
                    log_probs,
                    reduction='batchmean',
                    log_target=True
                )
                kl_penalty = kl_weight * kl_div
                total_kl_loss += kl_div.item()

                del ref_logits_shifted, ref_log_probs, kl_div

                # compute entropy on the fly
                probs = torch.nn.functional.softmax(shifted_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                entropy_bonus = -beta * entropy

                traj_loss = pg_loss + kl_penalty + entropy_bonus
                total_loss += traj_loss

                # cleanup memory
                del inputs, logits, shifted_logits, log_probs, token_ids
                del selected_log_probs, probs, entropy
                del pg_loss, kl_penalty, entropy_bonus, traj_loss

            torch.cuda.empty_cache()

        avg_loss = total_loss / len(trajectories)

        return avg_loss

    def save_checkpoint(self, path: str):
        self.save(path)


