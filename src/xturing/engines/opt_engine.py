from pathlib import Path
from typing import Optional, Union

from xturing.engines.causal import CausalEngine, CausalLoraEngine


class OPTEngine(CausalEngine):
    config_name: str = "opt_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(model_name="facebook/opt-1.3b", weights_path=weights_path)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class OPTLoraEngine(CausalLoraEngine):
    config_name: str = "opt_lora_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="facebook/opt-1.3b",
            weights_path=weights_path,
            target_modules=["q_proj", "v_proj"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class OPTInt8Engine(CausalEngine):
    config_name: str = "opt_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="facebook/opt-1.3b", weights_path=weights_path, load_8bit=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id


class OPTLoraInt8Engine(CausalLoraEngine):
    config_name: str = "opt_lora_int8_engine"

    def __init__(self, weights_path: Optional[Union[str, Path]] = None):
        super().__init__(
            model_name="facebook/opt-1.3b",
            weights_path=weights_path,
            load_8bit=True,
            target_modules=["q_proj", "v_proj"],
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
