from .model import Model
import torch
import torch.nn as nn
import fnmatch
from transformers import AutoTokenizer
from .hypernet.configuration_hypernet import HypernetConfig
from .hypernet.modeling_hypernet import HypernetModel
from torch.utils.data import DataLoader
from axbench.utils.data_utils import make_data_module


class HyperSteerWeight(Model):
    def __str__(self):
        return "HyperSteerWeight"

    def make_model(self, **kwargs):
        self.training_args = kwargs["model_params"]
        self.device = kwargs["device"]

        hypernet_name = self.training_args.hypernet_name_or_path
        self.hypernet_tokenizer = AutoTokenizer.from_pretrained(
            hypernet_name, model_max_length=512
        )
        self.hypernet_tokenizer.padding_side = "left"
        if self.hypernet_tokenizer.pad_token is None:
            self.hypernet_tokenizer.pad_token = self.hypernet_tokenizer.eos_token

        config = HypernetConfig.from_pretrained(
            hypernet_name,
            num_hidden_layers=self.training_args.num_hidden_layers,
            torch_dtype=torch.bfloat16,
            use_target_model_embedding=False,
        )

        if self.training_args.hypernet_initialize_from_pretrained:
            self.hypernet = HypernetModel.from_pretrained(hypernet_name, config=config)
        else:
            self.hypernet = HypernetModel(config=config)

        self.hypernet = self.hypernet.to(self.device, dtype=torch.bfloat16)

        d = self.model.config.hidden_size
        r = self.training_args.low_rank_dimension

        self.A = nn.Linear(d, r, bias=False).to(self.device)
        self.B = nn.Linear(r, d, bias=False).to(self.device)

        self.target_patterns = self.training_args.weight_target_modules

    def _get_targets(self):
        targets = []
        for name, p in self.model.named_parameters():
            for pat in self.target_patterns:
                if fnmatch.fnmatch(name, pat):
                    targets.append((name, p))
        return targets

    def train(self, examples, **kwargs):
        data_module = make_data_module(
            self.tokenizer, examples,
            concept_tokenizer=self.hypernet_tokenizer
        )

        dataloader = DataLoader(
            data_module["train_dataset"],
            batch_size=self.training_args.batch_size,
            collate_fn=data_module["data_collator"],
            shuffle=True,
        )

        optim = torch.optim.AdamW(
            list(self.hypernet.parameters()) +
            list(self.A.parameters()) +
            list(self.B.parameters()),
            lr=self.training_args.lr,
            weight_decay=self.training_args.weight_decay,
        )

        self.model.train()
        self.hypernet.train()

        for _ in range(self.training_args.n_epochs):
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                base_out = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_hidden_states=True,
                )

                base_hidden = base_out.hidden_states[self.layer]

                h = self.hypernet(
                    input_ids=batch["concept_input_ids"],
                    attention_mask=batch["concept_attention_mask"],
                    base_encoder_hidden_states=base_hidden,
                    base_encoder_attention_mask=batch["attention_mask"],
                ).last_hidden_state[:, -1]

                delta = self.B(self.A(h)).mean(dim=0)

                originals = {}
                for name, p in self._get_targets():
                    originals[name] = p.data.clone()
                    p.data += delta.view_as(p.data)

                loss = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                ).loss

                loss.backward()
                optim.step()
                optim.zero_grad()

                for name, p in self._get_targets():
                    p.data.copy_(originals[name])

    @torch.no_grad()
    def predict_steer(self, examples, **kwargs):
        self.model.eval()
        self.hypernet.eval()

        generations = []
        perplexities = []

        for _, row in examples.iterrows():
            inputs = self.tokenizer(
                row["input"], return_tensors="pt"
            ).to(self.device)

            concept = self.hypernet_tokenizer(
                row["input_concept"], return_tensors="pt"
            ).to(self.device)

            hidden = self.model(
                **inputs, output_hidden_states=True
            ).hidden_states[self.layer]

            h = self.hypernet(
                input_ids=concept["input_ids"],
                attention_mask=concept["attention_mask"],
                base_encoder_hidden_states=hidden,
                base_encoder_attention_mask=inputs["attention_mask"],
            ).last_hidden_state[:, -1]

            delta = self.B(self.A(h)).mean(dim=0)

            originals = {}
            for name, p in self._get_targets():
                originals[name] = p.data.clone()
                p.data += delta.view_as(p.data)

            out = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("eval_output_length", 128),
                do_sample=True,
                temperature=kwargs.get("temperature", 1.0),
            )

            gen = self.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            generations.append(gen)

            for name, p in self._get_targets():
                p.data.copy_(originals[name])

        return {
            "steered_generation": generations,
            "perplexity": perplexities,
            "strength": [1.0] * len(generations),
        }
