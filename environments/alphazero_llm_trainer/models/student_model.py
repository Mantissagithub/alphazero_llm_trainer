from typing import Optional
import torch
from torch.optim import AdamW
from unsloth import FastLanguageModel
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

        # Ensure learning_rate is a float
        if learning_rate is not None:
            self.learning_rate = float(learning_rate)
        else:
            self.learning_rate = 2e-5  # Default fallback

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

        print("ref model created")

    def update_ref_model(self):
        print("updating ref model for kl penalty")

        del self.ref_model
        torch.cuda.empty_cache()

        self._create_ref_model()

    def _load_model(self):
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
        kl_weight: float = 0.01 # kl penality weight
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

        for traj, advantage in zip(trajectories, advantages):
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

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            with torch.no_grad():
                ref_outputs = self.ref_model(**inputs)
                ref_logits = ref_outputs.logits
                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)

            token_ids = inputs['input_ids'][:, 1:]
            shifted_log_probs = log_probs[:, :-1, :]
            shifted_ref_log_probs = ref_log_probs[:, :-1, :]

            selected_log_probs = shifted_log_probs.gather(
                dim=-1,
                index=token_ids.unsqueeze(-1)
            ).squeeze(-1)

            pg_loss = -(selected_log_probs.sum() * advantage)

            probs = torch.nn.functional.softmax(logits[:, :-1, :], dim=-1)
            ref_probs = torch.nn.functional.softmax(ref_logits[:, :-1, :], dim=-1)

            kl_div = (probs * (shifted_log_probs - shifted_ref_log_probs)).sum()
            kl_penalty = kl_weight * kl_div

            total_kl_loss += kl_div.item()

            entropy = -(probs * shifted_log_probs).sum(dim=-1).mean()
            entropy_bonus = -beta * entropy

            traj_loss = pg_loss + kl_penalty + entropy_bonus
            total_loss += traj_loss

        avg_loss = total_loss / len(trajectories)

        return avg_loss

    def save_checkpoint(self, path: str):
        self.save(path)


