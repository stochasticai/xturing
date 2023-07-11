from typing import Any

from .data import get_jsonl
from .eval import Eval
from .metrics import get_accuracy
from .models import check_sampled_text
from .prompt import is_chat_prompt


class Match(Eval):
    def __init__(
        self,
        model_specs,
        samples_jsonl: str,
        *args,
        max_tokens: int = 500,
        num_few_shot: int = 0,
        few_shot_jsonl: str = None,
        **kwargs,
    ):
        super().__init__(model_specs, *args, **kwargs)
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

    def eval_sample(self, sample: Any, *_):
        prompt = sample["input"]
        if self.num_few_shot > 0:
            assert is_chat_prompt(sample["input"]), "few shot requires chat prompt"
            prompt = sample["input"][:-1]
            for s in self.few_shot[: self.num_few_shot]:
                prompt += s["sample"]
            prompt += sample["input"][-1:]

        return check_sampled_text(self.model_spec, prompt, expected=sample["ideal"])

    def run(self, recorder):
        samples = get_jsonl(self.samples_jsonl)
        self.eval_all_samples(recorder, samples)
        events = recorder.get_events("match")
        return {
            "accuracy": get_accuracy(events),
        }
