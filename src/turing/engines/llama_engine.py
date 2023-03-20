from pathlib import Path
from typing import Optional, Union

import evaluate
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM


class LLamaEngine:
    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        if weights_path is None:
            self.model = LlamaForCausalLM.from_pretrained(
                "decapoda-research/llama-7b-hf"
            )
        else:
            assert Path(
                weights_path
            ).is_dir(), "The weights path should be a existing directory"
            self.model = LlamaForCausalLM.from_pretrained(weights_path)

        self.loss_fct = nn.CrossEntropyLoss()

    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask", None),
        )

        if "label_mask" in batch:
            loss = self.loss_fct(
                outputs.get("logits"), batch["targets"], mask=batch["label_mask"]
            )
        else:
            loss = self.loss_fct(outputs.get("logits"), batch["targets"])

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