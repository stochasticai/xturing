from pathlib import Path
from typing import Optional, Union

import evaluate
import torch
import torch.nn as nn
from transformers import AutoTokenizer, GPTJForCausalLM

from turing.config import DEFAULT_DTYPE


class GPTJEngine:
    config_name: str = "gptj_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        if weights_path is None:
            self.model = GPTJForCausalLM.from_pretrained(
                "EleutherAI/gpt-j-6B", torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        else:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            self.model = GPTJForCausalLM.from_pretrained(
                weights_path, torch_dtype=DEFAULT_DTYPE
            )
            self.tokenizer = AutoTokenizer.from_pretrained(weights_path)

        self.loss_fct = nn.CrossEntropyLoss()

    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        if "label_mask" in batch:
            logits = outputs.get("logits").view(-1, outputs.get("logits").size(-1))
            targets = batch["targets"].view(-1)

            loss = self.loss_fct(logits, targets, mask=batch["label_mask"])
        else:
            logits = outputs.get("logits").view(-1, outputs.get("logits").size(-1))
            targets = batch["targets"].view(-1)
            loss = self.loss_fct(logits, targets)

        return loss

    def validation_step(self, batch):
        metrics = evaluate.load("accuracy")
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        logits = outputs.get("logits")
        preds = torch.argmax(logits, -1)
        acc = metrics.compute(preds, batch["targets"])

        return acc
