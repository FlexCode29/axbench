from pathlib import Path
import json
from .model import Model
import peft
from peft import PeftModel, LoraConfig, get_peft_model
import torch, einops
from tqdm.auto import tqdm
import os
import pandas as pd
from ..utils.constants import EXAMPLE_TAG
from torch.utils.data import DataLoader
from ..utils.model_utils import (
    set_decoder_norm_to_unit_norm,
    remove_gradient_parallel_to_decoder_directions,
    gather_residual_activations, 
    get_lr,
    calculate_l1_losses
)
from transformers import get_scheduler
from transformers import set_seed


class LoRA(Model):
    def __str__(self):
        return 'LoRA'

    @staticmethod
    def _slugify(text, max_len=64):
        if not text:
            return ""
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
        slug = "_".join([s for s in slug.split("_") if s])
        return slug[:max_len]
    
    def make_model(self, **kwargs):
        peft_config = LoraConfig(
            r=self.training_args.low_rank_dimension,
            lora_alpha=self.training_args.lora_alpha,
            target_modules=self.training_args.lora_components,
            layers_to_transform=self.training_args.lora_layers,
            use_rslora=True, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM"
        )
        ax_model = get_peft_model(self.model, peft_config)
        ax_model.to(self.device)
        ax_model.print_trainable_parameters()
        self.ax_model = ax_model
        # lora is concept-ful due to its nature.
        self.concept_id = kwargs.get("concept_id")
        self.concept_name = kwargs.get("concept_name")

    def save(self, dump_dir, **kwargs):
        # folder-based saving
        tag = kwargs.get("tag")
        concept_name = kwargs.get("concept_name", getattr(self, "concept_name", None))
        concept_slug = self._slugify(concept_name)
        dump_dir = Path(dump_dir) / "lora" / str(self.concept_id)
        if tag:
            tag_name = f"{tag}__{concept_slug}" if concept_slug else str(tag)
            dump_dir = dump_dir / tag_name
        dump_dir.mkdir(parents=True, exist_ok=True)
        self.ax_model.save_pretrained(dump_dir)
        if concept_name and not tag:
            concept_file = dump_dir / "concept_name.txt"
            concept_file.write_text(str(concept_name), encoding="utf-8")

    def load(self, dump_dir=None, **kwargs):
        # folder-based loading
        self.concept_id = kwargs.get("concept_id", getattr(self, "concept_id", None))
        self.concept_name = kwargs.get("concept_name", getattr(self, "concept_name", None))
        checkpoint_dir = kwargs.get("checkpoint_dir", None)
        is_trainable = kwargs.get("is_trainable", False)
        if checkpoint_dir is None:
            if self.concept_id is None:
                raise ValueError("concept_id is required to load LoRA checkpoint when checkpoint_dir is not provided.")
            dump_dir = Path(f"{dump_dir}/lora/{self.concept_id}")
        else:
            dump_dir = Path(checkpoint_dir)
        self.ax_model = PeftModel.from_pretrained(
            self.model, dump_dir, is_trainable=is_trainable)

    def train(self, examples, **kwargs):
        train_dataloader = self.make_dataloader(examples, **kwargs)
        torch.cuda.empty_cache()
        if "concept_name" in kwargs and kwargs["concept_name"]:
            self.concept_name = kwargs["concept_name"]
        use_wandb = getattr(self, "use_wandb", False)
        wandb_run = None
        if use_wandb:
            import wandb
            run_name = f"{self.concept_id}_{self.concept_name}" if self.concept_name else f"{self.concept_id}"
            if wandb.run is None:
                wandb_run = wandb.init(
                    project=kwargs.get("wandb_project", None),
                    entity=kwargs.get("wandb_name", None),
                    name=run_name,
                    dir="wandb",
                )
                wandb_run.config.update(
                    {
                        "concept_id": self.concept_id,
                        "concept_name": self.concept_name,
                        "model_name": kwargs.get("logging_metadata", {}).get("model_name", "LoRA"),
                        "layer": kwargs.get("logging_metadata", {}).get("layer", None),
                    },
                    allow_val_change=True,
                )
            else:
                wandb_run = wandb.run
        save_epochs = kwargs.get("save_epochs", None)
        if save_epochs is None:
            save_epochs = getattr(self.training_args, "save_epochs", None)
        if isinstance(save_epochs, int):
            save_epochs = {save_epochs}
        elif isinstance(save_epochs, (list, tuple, set)):
            save_epochs = {int(e) for e in save_epochs}
        else:
            save_epochs = set()

        # Optimizer and lr
        optimizer = torch.optim.AdamW(
            self.ax_model.parameters(), 
            lr=self.training_args.lr, weight_decay=self.training_args.weight_decay)
        steps_per_epoch = max(1, len(train_dataloader) // self.training_args.gradient_accumulation_steps)
        num_training_steps = self.training_args.n_epochs * steps_per_epoch
        start_epoch = int(kwargs.get("start_epoch", 0))
        resume_steps = start_epoch * steps_per_epoch
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        if resume_steps > 0:
            for _ in range(resume_steps):
                lr_scheduler.step()
        # Main training loop.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        progress_bar = tqdm(
            total=num_training_steps,
            initial=resume_steps,
            position=rank,
            leave=True,
        )
        curr_step = resume_steps
        
        did_step1_checkpoint = start_epoch > 0 or kwargs.get("resume_from") is not None
        for epoch in range(start_epoch, self.training_args.n_epochs):
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            for step, batch in enumerate(train_dataloader):
                # prepare input
                inputs = {k: v.to(self.device) for k, v in batch.items()}
        
                # forward
                outputs = self.ax_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=inputs["labels"]
                )
                
                # loss
                loss = outputs.loss
                loss = loss.mean()
                loss /= self.training_args.gradient_accumulation_steps
                # grads
                loss.backward()
                epoch_loss_sum += loss.detach().float().item()
                epoch_loss_count += 1

                if not did_step1_checkpoint:
                    save_root = kwargs.get("dump_dir", self.dump_dir)
                    if save_root is not None:
                        avg_loss = epoch_loss_sum / max(1, epoch_loss_count)
                        tag = "step_1"
                        self.save(save_root, tag=tag, concept_name=self.concept_name)
                        log_entry = {
                            "epoch": epoch + 1,
                            "step": step + 1,
                            "avg_loss": avg_loss,
                            "tag": tag,
                            "concept_id": self.concept_id,
                            "concept_name": self.concept_name,
                        }
                        log_path = Path(save_root) / "lora" / str(self.concept_id) / "loss_log.jsonl"
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(log_entry, ensure_ascii=True) + "\n")
                        ckpt_dir = Path(save_root) / "lora" / str(self.concept_id)
                        concept_slug = self._slugify(self.concept_name)
                        ckpt_dir = ckpt_dir / (f"{tag}__{concept_slug}" if concept_slug else tag)
                        (ckpt_dir / "loss.json").write_text(
                            json.dumps(log_entry, ensure_ascii=True, indent=2),
                            encoding="utf-8",
                        )
                        if use_wandb:
                            wandb.log(
                                {
                                    "checkpoint/avg_loss": avg_loss,
                                    "checkpoint/epoch": epoch + 1,
                                    "checkpoint/step": step + 1,
                                },
                                step=1,
                            )
                        did_step1_checkpoint = True

                # Perform optimization step every gradient_accumulation_steps
                if (step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    curr_step += 1
                    curr_lr = get_lr(optimizer)
                    # optim
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (curr_lr, loss))
                    if use_wandb:
                        wandb.log(
                            {
                                "train/loss": float(loss.detach().cpu().item()) * self.training_args.gradient_accumulation_steps,
                                "train/lr": float(curr_lr),
                                "train/epoch": epoch + 1,
                                "train/step": step + 1,
                            },
                            step=curr_step,
                        )
            if save_epochs and (epoch + 1) in save_epochs:
                save_root = kwargs.get("dump_dir", self.dump_dir)
                if save_root is not None:
                    avg_loss = epoch_loss_sum / max(1, epoch_loss_count)
                    tag = f"epoch_{epoch + 1}"
                    self.save(save_root, tag=tag, concept_name=self.concept_name)
                    # append loss log
                    log_entry = {
                        "epoch": epoch + 1,
                        "avg_loss": avg_loss,
                        "tag": tag,
                        "concept_id": self.concept_id,
                        "concept_name": self.concept_name,
                    }
                    log_path = Path(save_root) / "lora" / str(self.concept_id) / "loss_log.jsonl"
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry, ensure_ascii=True) + "\n")
                    # write loss alongside checkpoint
                    ckpt_dir = Path(save_root) / "lora" / str(self.concept_id)
                    concept_slug = self._slugify(self.concept_name)
                    ckpt_dir = ckpt_dir / (f"{tag}__{concept_slug}" if concept_slug else tag)
                    (ckpt_dir / "loss.json").write_text(
                        json.dumps(log_entry, ensure_ascii=True, indent=2),
                        encoding="utf-8",
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "checkpoint/avg_loss": avg_loss,
                                "checkpoint/epoch": epoch + 1,
                            },
                            step=epoch + 1,
                        )
        progress_bar.close()
        if use_wandb and wandb_run is not None:
            wandb_run.finish()

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.ax_model.eval()
        # set tokenizer padding to left
        self.tokenizer.padding_side = "left"

        # iterate rows in batch
        batch_size = kwargs.get("batch_size", 64)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)
        all_generations = []
        all_perplexities = []
        # Main training loop.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        progress_bar = tqdm(range(0, len(examples), batch_size), position=rank, leave=True)
        for i in range(0, len(examples), batch_size):
            batch_examples = examples.iloc[i:i+batch_size]
            input_strings = batch_examples['input'].tolist()
            # tokenize input_strings
            inputs = self.tokenizer(
                input_strings, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            generations = self.ax_model.generate(
                **inputs, 
                max_new_tokens=eval_output_length, do_sample=True, 
                temperature=temperature,
            )

            # Decode and print only the generated text without prompt tokens
            input_lengths = [len(input_ids) for input_ids in inputs.input_ids]
            generated_texts = [
                self.tokenizer.decode(generation[input_length:], skip_special_tokens=True)
                for generation, input_length in zip(generations, input_lengths)
            ]
            all_generations += generated_texts

            # Calculate perplexity for each sequence
            unpruned_generated_texts = [
                self.tokenizer.decode(generation, skip_special_tokens=True)
                for generation in generations
            ]
            batch_input_ids = self.tokenizer(
                unpruned_generated_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
            batch_attention_mask = (batch_input_ids != self.tokenizer.pad_token_id).float()
            
            # Forward pass without labels to get logits
            outputs = self.model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            
            logits = outputs.logits[:, :-1, :].contiguous()  # Remove last token prediction
            target_ids = batch_input_ids[:, 1:].contiguous()  # Shift right by 1
            
            # Calculate loss for each token
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            
            # Reshape losses and mask
            token_losses = token_losses.view(batch_input_ids.size(0), -1)
            mask = batch_attention_mask[:, 1:].contiguous()
            
            # Calculate perplexity for each sequence
            seq_lengths = mask.sum(dim=1)
            seq_losses = (token_losses * mask).sum(dim=1) / seq_lengths
            seq_perplexities = torch.exp(seq_losses).tolist()
            all_perplexities.extend(seq_perplexities)
            progress_bar.update(1)

        return {
            "steered_generation": all_generations,
            "perplexity": all_perplexities,
        }
