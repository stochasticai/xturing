from typing import Optional

from xturing.engines.gpt2_engine import (
    GPT2Engine,
    GPT2Int8Engine,
    GPT2LoraEngine,
    GPT2LoraInt8Engine,
)
from xturing.models.causal import (
    CausalInt8Model,
    CausalLoraInt8Model,
    CausalLoraModel,
    CausalModel,
)


class GPT2(CausalModel):
    config_name: str = "gpt2"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPT2Engine.config_name, weights_path)


class GPT2Lora(CausalLoraModel):
    config_name: str = "gpt2_lora"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPT2LoraEngine.config_name, weights_path)


class GPT2Int8(CausalInt8Model):
    config_name: str = "gpt2_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPT2Int8Engine.config_name, weights_path)


class GPT2LoraInt8(CausalLoraInt8Model):
    config_name: str = "gpt2_lora_int8"

    def __init__(self, weights_path: Optional[str] = None):
        super().__init__(GPT2LoraInt8Engine.config_name, weights_path)
