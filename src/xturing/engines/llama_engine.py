import os
from pathlib import Path
from typing import Optional, Union

import torch
import transformers
from torch import nn

from xturing.engines.causal import CausalEngine, CausalLoraEngine
from xturing.engines.llama_utils import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from xturing.engines.lora_engine import prepare_model_for_int8_training
from xturing.engines.quant_utils import autotune_warmup, make_quant
from xturing.utils.hub import ModelHub


class LLamaEngine(CausalEngine):
    config_name: str = "llama_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(weights_path=weights_path, model=model, tokenizer=tokenizer)

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraEngine(CausalLoraEngine):
    config_name: str = "llama_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path,
            model=model,
            tokenizer=tokenizer,
            target_modules=["q_proj", "v_proj"],
        )


class LLamaInt8Engine(CausalEngine):
    config_name: str = "llama_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device_map,
        )
        model = prepare_model_for_int8_training(model)
        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path, model=model, tokenizer=tokenizer, load_8bit=True
        )

    def save(self, saving_path: Union[str, Path]):
        self.model.save_pretrained(saving_path)
        self.tokenizer.save_pretrained(saving_path)


class LlamaLoraInt8Engine(CausalLoraEngine):
    config_name: str = "llama_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "aleksickx/llama-7b-hf"
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map=device_map,
        )
        model = prepare_model_for_int8_training(model)

        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            weights_path=weights_path,
            model=model,
            tokenizer=tokenizer,
            load_8bit=True,
            target_modules=["q_proj", "v_proj"],
        )


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


class LlamaLoraInt4Engine(CausalLoraEngine):
    config_name: str = "llama_lora_int4_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        model_name = "decapoda-research/llama-7b-hf"

        if weights_path is None:
            weights_path = ModelHub().load("x/llama_lora_int4")

        config = LlamaConfig.from_pretrained(model_name)

        saved_kaiming_uniform_ = torch.nn.init.kaiming_uniform_
        saved_uniform_ = torch.nn.init.uniform_
        saved_normal_ = torch.nn.init.normal_

        def noop(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        torch.set_default_dtype(torch.half)
        model = LlamaForCausalLM(config)
        torch.set_default_dtype(torch.float)
        model = model.eval()

        layers = find_layers(model)

        for name in ["lm_head"]:
            if name in layers:
                del layers[name]

        wbits = 4
        groupsize = 128
        warmup_autotune = True

        make_quant(model, layers, wbits, groupsize)

        state_dict = torch.load(
            weights_path / Path("pytorch_model.bin"), map_location="cpu"
        )

        if warmup_autotune:
            autotune_warmup(model)

        model.seqlen = 2048

        model.gptq = True

        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        tokenizer = LlamaTokenizer.from_pretrained(model_name, add_bos_token=False)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
        )

        torch.nn.init.kaiming_uniform_ = saved_kaiming_uniform_
        torch.nn.init.uniform_ = saved_uniform_
        torch.nn.init.normal_ = saved_normal_

        self.set_from_state_dict(state_dict)
