import json
import os
from collections import OrderedDict
from contextlib import contextmanager
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math
from statistics import pstdev

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from transformers import AutoTokenizer

from .hypernet.configuration_hypernet import HypernetConfig
from .hypernet.modeling_hypernet import HypernetModel
from .model import Model
from ..utils.data_utils import make_data_module
from torch.optim.lr_scheduler import LambdaLR


METADATA_FILE = "metadata.jsonl"


class HyperSteerWeightLinear(nn.Module):
    """
    Wraps an existing Linear layer with a concept-conditioned low-rank update.
    """

    def __init__(self, base_module: nn.Linear, rank: int, dtype: torch.dtype):
        super().__init__()
        self.rank = rank
        self.linear = base_module
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.left_basis = nn.Parameter(
            torch.empty(rank, base_module.out_features, dtype=dtype)
        )
        self.right_basis = nn.Parameter(
            torch.empty(rank, base_module.in_features, dtype=dtype)
        )
        
        # LoRA-style initialization
        # Left = Kaiming, Right = Zero -> Starts as Identity function
        nn.init.kaiming_uniform_(self.left_basis, a=math.sqrt(5))
        nn.init.zeros_(self.right_basis)

        self.current_coeffs: Optional[torch.Tensor] = None

    def set_coeffs(self, coeffs: torch.Tensor):
        if coeffs.dim() == 1:
            coeffs = coeffs.unsqueeze(0)
        if coeffs.dim() != 2:
            raise ValueError("Concept coefficients must be 1D or 2D.")
        self.current_coeffs = coeffs.to(self.left_basis.device, dtype=self.left_basis.dtype)

    def clear_concept(self):
        self.current_coeffs = None

    def get_adapter_state(self) -> Dict[str, torch.Tensor]:
        return {
            "left_basis": self.left_basis.detach().cpu(),
            "right_basis": self.right_basis.detach().cpu(),
        }

    def load_adapter_state(self, state: Dict[str, torch.Tensor]):
        self.left_basis.data.copy_(state["left_basis"].to(device=self.left_basis.device, dtype=self.left_basis.dtype))
        self.right_basis.data.copy_(state["right_basis"].to(device=self.right_basis.device, dtype=self.right_basis.dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(hidden_states)
        coeffs = self.current_coeffs
        if coeffs is None:
            return base_out
        
        # Optimization: If coeffs are zero, skip computation
        if torch.all(coeffs == 0):
            return base_out

        batch_size = hidden_states.shape[0]
        if coeffs.shape[0] == 1 and batch_size != 1:
            coeffs = coeffs.expand(hidden_states.shape[0], -1)
        
        if coeffs.shape[0] != batch_size:
             coeffs = coeffs[:1].expand(batch_size, -1)

        last_dim = hidden_states.shape[-1]
        reshaped = hidden_states.reshape(batch_size, -1, last_dim)
        proj = torch.matmul(reshaped, self.right_basis.t()) 
        scaled = proj * coeffs.unsqueeze(1)
        delta = torch.matmul(scaled, self.left_basis)
        delta = delta.reshape_as(base_out)
        return base_out + delta


class HyperSteerWeight(Model):
    """Hypernetwork that produces low-rank weight updates for specific LM layers."""

    def __str__(self):
        return "HyperSteerWeight"

    def make_model(self, **kwargs):
        default_dtype = None
        if hasattr(self.model, "dtype") and self.model.dtype is not None:
            default_dtype = self.model.dtype
        else:
            try:
                default_dtype = next(self.model.parameters()).dtype
            except StopIteration:
                default_dtype = torch.float32
        dtype = kwargs.get("dtype", default_dtype)
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self.low_rank_dimension = kwargs.get(
            "low_rank_dimension",
            getattr(self.training_args, "low_rank_dimension", 1),
        )

        pretrained_state_dir = kwargs.get("pretrained_state_dir")
        saved_state = None
        if pretrained_state_dir:
            state_path = Path(pretrained_state_dir) / "adapter_state.pt"
            if state_path.exists():
                saved_state = torch.load(state_path, map_location="cpu")
                self.low_rank_dimension = saved_state.get(
                    "low_rank_dimension", self.low_rank_dimension
                )

        model_params = kwargs.get("model_params", self.training_args)
        target_patterns = getattr(model_params, "weight_target_modules", None)
        if not target_patterns:
            raise ValueError("weight_target_modules must be provided for HyperSteerWeight.")
        self.target_patterns = target_patterns

        self.metadata_path = kwargs.get(
            "metadata_path",
            os.path.join(kwargs.get("dump_dir", ""), METADATA_FILE),
        )
        self._load_metadata()

        self.hyper_inputs_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.base_token_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self.base_hidden_cache: OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]] = OrderedDict()
        self.base_hidden_cache_size = kwargs.get("hidden_cache_size", 128)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        hypernet_name = kwargs.get(
            "hypernet_name_or_path",
            getattr(model_params, "hypernet_name_or_path", None) or self.training_args.model_name,
        )

        self.hypernet_tokenizer = AutoTokenizer.from_pretrained(hypernet_name, model_max_length=512)
        self.hypernet_tokenizer.padding_side = "left"
        if self.hypernet_tokenizer.pad_token is None:
            self.hypernet_tokenizer.pad_token = self.hypernet_tokenizer.eos_token
            self.hypernet_tokenizer.pad_token_id = self.hypernet_tokenizer.eos_token_id

        num_hidden_layers = kwargs.get(
            "num_hidden_layers",
            getattr(model_params, "num_hidden_layers", None),
        )

        hypernet_config = HypernetConfig.from_pretrained(
            pretrained_model_name_or_path=hypernet_name,
            num_hidden_layers=num_hidden_layers,
            torch_dtype=torch.bfloat16 if dtype == torch.bfloat16 else dtype,
            use_target_model_embedding=False,
        )

        use_pretrained_hnet = kwargs.get(
            "hypernet_initialize_from_pretrained",
            getattr(model_params, "hypernet_initialize_from_pretrained", True),
        )
        if use_pretrained_hnet:
            self.concept_embedding = HypernetModel.from_pretrained(
                hypernet_name, config=hypernet_config
            )
        else:
            self.concept_embedding = HypernetModel(config=hypernet_config)

        self.concept_embedding = self.concept_embedding.to(self.device, dtype=dtype)
        self.concept_embedding.gradient_checkpointing_disable()

        self.adapters: "OrderedDict[str, HyperSteerWeightLinear]" = OrderedDict()
        self._wrap_target_modules(dtype)

        hidden_dim = self.concept_embedding.config.hidden_size
        self.hyper_projector = nn.Linear(
            hidden_dim, len(self.adapters) * self.low_rank_dimension, bias=True
        ).to(self.device, dtype=dtype)

        if saved_state:
            self.hyper_projector.load_state_dict(saved_state["hyper_projector"])
            saved_adapters = saved_state.get("adapters", {})
            for name, adapter in self.adapters.items():
                if name in saved_adapters:
                    adapter.load_adapter_state(saved_adapters[name])

    def _wrap_target_modules(self, dtype: torch.dtype):
        modules_to_wrap: List[str] = []
        for name, module in self.model.named_modules():
            param_name = f"{name}.weight" if name else "weight"
            if any(fnmatch(param_name, pattern) for pattern in self.target_patterns):
                if not isinstance(module, nn.Linear):
                    if isinstance(module, HyperSteerWeightLinear):
                         self.adapters[name] = module
                         continue
                    raise ValueError(f"Target module {name} is not nn.Linear.")
                modules_to_wrap.append(name)

        if not modules_to_wrap and not self.adapters:
            raise ValueError("No modules matched weight_target_modules patterns.")

        for name in modules_to_wrap:
            parent, attr = self._locate_parent_module(name)
            base_module = getattr(parent, attr)
            adapter = HyperSteerWeightLinear(base_module, self.low_rank_dimension, dtype).to(self.device)
            setattr(parent, attr, adapter)
            self.adapters[name] = adapter

    def _locate_parent_module(self, module_name: str) -> Tuple[nn.Module, str]:
        parts = module_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent, parts[-1]

    def _load_metadata(self):
        self.concept_id_to_text: Dict[int, str] = {}
        if not self.metadata_path or not os.path.exists(self.metadata_path):
            raise ValueError(f"Metadata file not found at {self.metadata_path}.")
        with open(self.metadata_path, "r") as f:
            for line in f:
                data = json.loads(line)
                self.concept_id_to_text[data["concept_id"]] = data["concept"]

    def make_dataloader(self, examples, rank, world_size, shuffle=True, distributed=False, concept_tokenizer=None, **kwargs):
        if distributed:
            sampler = DistributedSampler(
                examples,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
            )
            data_module = make_data_module(
                self.tokenizer,
                examples,
                concept_tokenizer=self.hypernet_tokenizer,
                **kwargs,
            )
            g = torch.Generator()
            g.manual_seed(self.seed)
            train_dataloader = DataLoader(
                data_module["train_dataset"],
                batch_size=self.training_args.batch_size,
                collate_fn=data_module["data_collator"],
                sampler=sampler,
                generator=g,
            )
            return train_dataloader, sampler
        data_module = make_data_module(
            self.tokenizer,
            examples,
            concept_tokenizer=self.hypernet_tokenizer,
            **kwargs,
        )
        g = torch.Generator()
        g.manual_seed(self.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"],
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
            shuffle=shuffle,
            generator=g,
        )
        return train_dataloader

    def _cache_base_hidden(self, concept_id: int, hidden: torch.Tensor, mask: torch.Tensor):
        if self.base_hidden_cache_size <= 0:
            return
        if concept_id in self.base_hidden_cache:
            self.base_hidden_cache.move_to_end(concept_id)
        self.base_hidden_cache[concept_id] = (hidden.cpu(), mask.cpu())
        if len(self.base_hidden_cache) > self.base_hidden_cache_size:
            self.base_hidden_cache.popitem(last=False)

    def _get_concept_base_inputs(self, concept_id: int) -> Dict[str, torch.Tensor]:
        if concept_id not in self.base_token_cache:
            text = self.concept_id_to_text[concept_id]
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=True,
            )
            self.base_token_cache[concept_id] = {k: v for k, v in tokens.items()}
        cached = self.base_token_cache[concept_id]
        return {k: v.clone().to(self.device) for k, v in cached.items()}

    def _get_concept_hidden_states(self, concept_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if concept_id in self.base_hidden_cache:
            hidden, mask = self.base_hidden_cache.pop(concept_id)
            self.base_hidden_cache[concept_id] = (hidden, mask)
            return hidden.to(self.device, dtype=self.dtype), mask.to(self.device)
        inputs = self._get_concept_base_inputs(concept_id)
        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[self.layer].detach()
        self._cache_base_hidden(concept_id, hidden, inputs["attention_mask"].detach())
        return hidden.to(self.device, dtype=self.dtype), inputs["attention_mask"].to(self.device)

    def _get_hyper_inputs(self, concept_id: int) -> Dict[str, torch.Tensor]:
        if concept_id not in self.hyper_inputs_cache:
            text = self.concept_id_to_text[concept_id]
            tokens = self.hypernet_tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            self.hyper_inputs_cache[concept_id] = {k: v for k, v in tokens.items()}
        cached = self.hyper_inputs_cache[concept_id]
        return {k: v.clone().to(self.device) for k, v in cached.items()}

    def _pool_hidden(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom

    def _compute_concept_coeffs(self, concept_id: int, embedding_model=None) -> torch.Tensor:
        model = embedding_model if embedding_model is not None else self.concept_embedding
        hyper_inputs = self._get_hyper_inputs(concept_id)
        base_hidden, base_mask = self._get_concept_hidden_states(concept_id)
        outputs = model(
            input_ids=hyper_inputs["input_ids"],
            attention_mask=hyper_inputs["attention_mask"],
            base_encoder_hidden_states=base_hidden,
            base_encoder_attention_mask=base_mask,
            output_hidden_states=False,
        ).last_hidden_state
        pooled = self._pool_hidden(outputs, hyper_inputs["attention_mask"])
        logits = self.hyper_projector(pooled)
        coeffs = logits.view(-1, len(self.adapters), self.low_rank_dimension)
        
        # MODIFIED: Scaled Tanh to allow larger dynamic range.
        # This gives the model 10x leverage to "shout" over the base model weights.
        return torch.tanh(coeffs) * 10.0

    @contextmanager
    def _activate_coefficients(self, coeffs: torch.Tensor):
        if coeffs.dim() == 2:
            coeffs = coeffs.unsqueeze(0)
        if coeffs.dim() != 3 or coeffs.shape[1] != len(self.adapters):
            raise ValueError("Coefficient tensor must have shape [batch, num_modules, rank].")
        
        for idx, adapter in enumerate(self.adapters.values()):
            adapter.set_coeffs(coeffs[:, idx, :])
        try:
            yield
        finally:
            for adapter in self.adapters.values():
                adapter.clear_concept()

    def _build_batch_coeffs(self, concept_ids: List[int], embedding_model=None) -> torch.Tensor:
        coeffs = []
        for concept_id in concept_ids:
            coeff = self._compute_concept_coeffs(concept_id, embedding_model=embedding_model)[0]
            coeffs.append(coeff)
        stacked = torch.stack(coeffs, dim=0)
        return stacked.to(self.device, dtype=self.dtype)
        
    def _sync_gradients(self, params: List[nn.Parameter], world_size: int):
        if world_size <= 1:
            return
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                p.grad.data /= world_size

    def _log_generations(self, batch, embedding_model, step, wandb_run):
        """Samples 1 example, generates steered vs unsteered, and logs to WandB."""
        # 1. Aggressive Cleanup to prevent OOM
        torch.cuda.empty_cache()
        
        # 2. Unwrap DDP Model (Critical for .generate memory overhead)
        # Using .module prevents duplicating the wrapper overhead
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # 3. Process only ONE example to save memory
        input_ids = batch["input_ids"][:1] 
        concept_ids = batch["concept_ids"][:1].detach().cpu().tolist()
        
        prompts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        # Extract text before "\nmodel\n"
        prompts = [p.split("\nmodel\n", 1)[0] for p in prompts]

        concepts = [self.concept_id_to_text.get(cid, str(cid)) for cid in concept_ids]
        
        # 4. Inference Mode (Saves more memory than no_grad)
        with torch.inference_mode():
            # Base Gen
            base_gens, base_ppls = self._generate(prompts, max_new_tokens=32, temperature=0.7)
            
            # Steered Gen
            coeffs = self._build_batch_coeffs(concept_ids, embedding_model=embedding_model)
            with self._activate_coefficients(coeffs):
                # We reuse the helper _generate but we need to ensure it doesn't leak.
                # Actually, calling raw_model.generate directly is safer here to avoid
                # any hook overheads from the wrapper class if they exist.
                
                inputs = self._prepare_generation(prompts)
                outputs = raw_model.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.7)
                steered_gens = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
                # Skip detailed PPL calc for speed/memory if OOM is tight during logging
                steered_ppls = [0.0] 

        # 5. Log
        import wandb
        columns = ["Step", "Concept", "Input", "Base Gen", "Steered Gen"]
        data = [[step, concepts[0], prompts[0], base_gens[0], steered_gens[0]]]
        
        wandb_run.log({"eval/completions": wandb.Table(columns=columns, data=data)}, step=step)
        
        # 6. Final Cleanup
        del raw_model, coeffs, inputs, outputs, steered_gens, base_gens
        torch.cuda.empty_cache()

    def train(self, examples, **kwargs):
        if not dist.is_initialized():
            raise RuntimeError("HyperSteerWeight requires torch.distributed to be initialized.")
        rank = dist.get_rank()
        world_size = kwargs.get("world_size", dist.get_world_size())
        kwargs = dict(kwargs)
        kwargs.pop("world_size", None)
        log_to_wandb = self.use_wandb and rank == 0
        wandb_run = None
        if log_to_wandb:
            import wandb
            run_name = kwargs.get("run_name") or f"{self.__str__()}_{self.layer}"
            wandb_init_kwargs = {"dir": "wandb", "name": run_name}
            if kwargs.get("wandb_project"): wandb_init_kwargs["project"] = kwargs.get("wandb_project")
            if kwargs.get("wandb_name"): wandb_init_kwargs["entity"] = kwargs.get("wandb_name")
            wandb_run = wandb.init(**wandb_init_kwargs)

        train_dataloader, train_sampler = self.make_dataloader(
            examples,
            rank=rank,
            world_size=world_size,
            concept_tokenizer=self.hypernet_tokenizer,
            distributed=True,
            **kwargs,
        )

        embedding_model = (
            self.concept_embedding
            if world_size == 1
            else DDP(self.concept_embedding, device_ids=[rank], find_unused_parameters=True)
        )

        # Explicitly select ONLY trainable params
        trainable_params: List[nn.Parameter] = list(embedding_model.parameters())
        unwrapped_params: List[nn.Parameter] = list(self.hyper_projector.parameters())
        for adapter in self.adapters.values():
            unwrapped_params.append(adapter.left_basis)
            unwrapped_params.append(adapter.right_basis)
        trainable_params += unwrapped_params

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.training_args.lr,
            weight_decay=self.training_args.weight_decay,
        )
        steps_per_epoch = max(1, math.ceil(len(train_dataloader) / self.training_args.gradient_accumulation_steps))
        num_training_steps = self.training_args.n_epochs * steps_per_epoch
        lr_scheduler = self._build_lr_scheduler(optimizer, num_training_steps)

        progress_bar = tqdm(range(num_training_steps), position=rank, leave=True)
        embedding_model.train()
        self.hyper_projector.train()

        optimizer.zero_grad()
        global_step = 0
        loss_history: List[float] = []
        
        # Loss function setup for Standard CE Loss
        ce_loss_fct = nn.CrossEntropyLoss(reduction='none')
        
        eval_every_steps = kwargs.get("eval_every_steps", 50)

        try:
            for epoch in range(self.training_args.n_epochs):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                for step, batch in enumerate(train_dataloader):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    concept_id_list = batch["concept_ids"].detach().cpu().tolist()
                    
                    coeff_tensor = self._build_batch_coeffs(concept_id_list, embedding_model=embedding_model)
                    
                    with self._activate_coefficients(coeff_tensor):
                        outputs = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                    
                    # --- STANDARD CE LOSS ---
                    logits = outputs.logits
                    labels = batch["labels"]
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                    ce_loss = ce_loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1)
                    )
                    
                    active_loss = shift_labels.view(-1) != -100
                    
                    if active_loss.sum() > 0:
                        raw_loss = ce_loss[active_loss].mean()
                        with torch.no_grad():
                            true_ppl_val = torch.exp(raw_loss).item()
                    else:
                        raw_loss = torch.tensor(0.0).to(self.device)
                        true_ppl_val = 0.0
                    
                    loss = raw_loss / self.training_args.gradient_accumulation_steps
                    loss.backward()

                    # CRITICAL: Delete graph before stepping optimizer to free max memory
                    del logits, shift_logits, outputs, ce_loss
                    
                    should_step = ((step + 1) % self.training_args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader))
                    if should_step:
                        self._sync_gradients(unwrapped_params, world_size)
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        
                        optimizer.step()
                        lr_scheduler.step()
                        # set_to_none=True releases memory faster than zero_grad()
                        optimizer.zero_grad(set_to_none=True)
                        
                        progress_bar.update(1)
                        global_step += 1
                        
                        # Logging Stats
                        current_lr = optimizer.param_groups[0]["lr"]
                        loss_val_tensor = raw_loss.detach()
                        dist.all_reduce(loss_val_tensor, op=dist.ReduceOp.SUM)
                        raw_loss_val = (loss_val_tensor / world_size).item()
                        loss_history.append(raw_loss_val)
                        progress_bar.set_description(f"rank {rank} loss {raw_loss_val:.4f} lr {current_lr:.6f}")
                        
                        if log_to_wandb and wandb_run is not None:
                            wandb.log({
                                "train/loss": raw_loss_val,
                                "train/perplexity": true_ppl_val,
                                "train/learning_rate": current_lr
                            }, step=global_step)
                            
                            # GENERATION LOGGING (Clean-Room Approach)
                            # Run only when GPU is absolutely empty after zero_grad
                            if global_step % eval_every_steps == 0 or global_step == 1:
                                self._log_generations(batch, embedding_model, global_step, wandb_run)

        finally:
            progress_bar.close()
            if wandb_run is not None:
                wandb_run.finish()

    def _build_lr_scheduler(self, optimizer, num_training_steps: int) -> LambdaLR:
        total_steps = max(1, num_training_steps)
        decay_start = getattr(self.training_args, "lr_decay_start_step", 0) or 0
        decay_start = max(0, min(decay_start, total_steps))
        base_lr = self.training_args.lr or 0.0
        min_lr = getattr(self.training_args, "lr_min", 0.0) or 0.0
        if base_lr <= 0:
            min_factor = 1.0
        else:
            min_lr = max(0.0, min(min_lr, base_lr))
            min_factor = min_lr / base_lr if base_lr > 0 else 0.0
        if decay_start >= total_steps or min_factor == 1.0:
            return LambdaLR(optimizer, lambda step: 1.0)
        decay_range = max(1, total_steps - decay_start)
        def lr_lambda(step: int) -> float:
            if step <= decay_start: return 1.0
            decay_progress = min(step - decay_start, decay_range)
            return 1.0 - (1.0 - min_factor) * (decay_progress / decay_range)
        return LambdaLR(optimizer, lr_lambda)

    def _prepare_generation(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        prev_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        encoded = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        self.tokenizer.padding_side = prev_padding
        return encoded

    def _generate(self, prompts: List[str], max_new_tokens: int, temperature: float) -> Tuple[List[str], torch.Tensor]:
        inputs = self._prepare_generation(prompts)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature)
        input_lengths = [len(ids) for ids in inputs.input_ids]
        generations = [self.tokenizer.decode(out[input_len:], skip_special_tokens=True) for out, input_len in zip(outputs, input_lengths)]
        
        # PPL calc on generated text
        decoded_full = [self.tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
        gen_tokens = self.tokenizer(decoded_full, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            model_outputs = self.model(input_ids=gen_tokens.input_ids, attention_mask=gen_tokens.attention_mask)
        logits = model_outputs.logits[:, :-1, :].contiguous()
        targets = gen_tokens.input_ids[:, 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss(reduction="none")
        token_losses = loss_fct(logits.view(-1, logits.size(-1)), targets.view(-1)).view(targets.size())
        mask = gen_tokens.attention_mask[:, 1:]
        perplexities = torch.exp((token_losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1))
        return generations, perplexities.detach().cpu()

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.model.eval()
        results_generations = [""] * len(examples)
        results_perplexity = [0.0] * len(examples)
        batch_size = kwargs.get("batch_size", 8)
        eval_output_length = kwargs.get("eval_output_length", 128)
        temperature = kwargs.get("temperature", 1.0)

        examples = examples.reset_index(drop=True)
        concept_ids = examples["concept_id"].unique().tolist()
        for concept_id in concept_ids:
            concept_df = examples[examples["concept_id"] == concept_id]
            coeffs = self._compute_concept_coeffs(concept_id)[0].unsqueeze(0).to(self.device, dtype=self.dtype)
            factors = concept_df["factor"].unique().tolist() if "factor" in concept_df else [1.0]
            for factor in factors:
                factor_df = concept_df[concept_df["factor"] == factor] if "factor" in concept_df else concept_df
                idxs = factor_df.index.tolist()
                for start in range(0, len(idxs), batch_size):
                    batch_indices = idxs[start : start + batch_size]
                    prompts = factor_df.loc[batch_indices, "input"].tolist()
                    scaled_coeffs = coeffs * factor
                    
                    if factor == 0:
                        generations, perplexities = self._generate(prompts, max_new_tokens=eval_output_length, temperature=temperature)
                    else:
                        with self._activate_coefficients(scaled_coeffs):
                            generations, perplexities = self._generate(prompts, max_new_tokens=eval_output_length, temperature=temperature)
                            
                    for local_idx, text in zip(batch_indices, generations):
                        results_generations[local_idx] = text
                    for local_idx, ppl in zip(batch_indices, perplexities.tolist()):
                        results_perplexity[local_idx] = ppl

        return {
            "steered_generation": results_generations,
            "perplexity": results_perplexity,
        }

    def save(self, dump_dir, **kwargs):
        dump_dir = Path(dump_dir)
        weight_dir = dump_dir / "hypersteer_weight"
        weight_dir.mkdir(parents=True, exist_ok=True)
        self.concept_embedding.save_pretrained(weight_dir)
        self.hypernet_tokenizer.save_pretrained(weight_dir)
        adapter_state = {
            "hyper_projector": self.hyper_projector.state_dict(),
            "adapters": {name: adapter.get_adapter_state() for name, adapter in self.adapters.items()},
            "low_rank_dimension": self.low_rank_dimension,
        }
        torch.save(adapter_state, weight_dir / "adapter_state.pt")

    def load(self, dump_dir=None, **kwargs):
        weight_dir = Path(dump_dir) / "hypersteer_weight"
        kwargs["hypernet_name_or_path"] = str(weight_dir)
        kwargs["pretrained_state_dir"] = str(weight_dir)
        if "metadata_path" not in kwargs:
            kwargs["metadata_path"] = os.path.join(dump_dir, METADATA_FILE)
        self.make_model(**kwargs)