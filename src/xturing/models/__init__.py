from xturing.models.base import BaseModel
from xturing.models.bloom import Bloom, BloomInt8, BloomLora, BloomLoraInt8
from xturing.models.cerebras import (
    Cerebras,
    CerebrasInt8,
    CerebrasLora,
    CerebrasLoraInt8,
)
from xturing.models.distilgpt2 import DistilGPT2, DistilGPT2Lora
from xturing.models.falcon import Falcon, FalconInt8, FalconLora, FalconLoraInt8
from xturing.models.galactica import (
    Galactica,
    GalacticaInt8,
    GalacticaLora,
    GalacticaLoraInt8,
)
from xturing.models.generic import (
    GenericInt8Model,
    GenericLoraInt8Model,
    GenericLoraModel,
    GenericModel,
)
from xturing.models.gpt2 import GPT2, GPT2Int8, GPT2Lora, GPT2LoraInt8
from xturing.models.gptj import GPTJ, GPTJInt8, GPTJLora, GPTJLoraInt8
from xturing.models.llama import (
    Llama,
    LlamaInt8,
    LlamaLora,
    LlamaLoraInt4,
    LlamaLoraInt8,
)
from xturing.models.opt import OPT, OPTInt8, OPTLora, OPTLoraInt8
from xturing.models.stable_diffusion import StableDiffusion

BaseModel.add_to_registry(DistilGPT2.config_name, DistilGPT2)
BaseModel.add_to_registry(DistilGPT2Lora.config_name, DistilGPT2Lora)
BaseModel.add_to_registry(GPT2.config_name, GPT2)
BaseModel.add_to_registry(GPT2Lora.config_name, GPT2Lora)
BaseModel.add_to_registry(GPT2Int8.config_name, GPT2Int8)
BaseModel.add_to_registry(GPT2LoraInt8.config_name, GPT2LoraInt8)
BaseModel.add_to_registry(GPTJ.config_name, GPTJ)
BaseModel.add_to_registry(GPTJLora.config_name, GPTJLora)
BaseModel.add_to_registry(GPTJInt8.config_name, GPTJInt8)
BaseModel.add_to_registry(GPTJLoraInt8.config_name, GPTJLoraInt8)
BaseModel.add_to_registry(Llama.config_name, Llama)
BaseModel.add_to_registry(LlamaLora.config_name, LlamaLora)
BaseModel.add_to_registry(LlamaInt8.config_name, LlamaInt8)
BaseModel.add_to_registry(LlamaLoraInt8.config_name, LlamaLoraInt8)
BaseModel.add_to_registry(LlamaLoraInt4.config_name, LlamaLoraInt4)
BaseModel.add_to_registry(Galactica.config_name, Galactica)
BaseModel.add_to_registry(GalacticaLora.config_name, GalacticaLora)
BaseModel.add_to_registry(GalacticaInt8.config_name, GalacticaInt8)
BaseModel.add_to_registry(GalacticaLoraInt8.config_name, GalacticaLoraInt8)
BaseModel.add_to_registry(OPT.config_name, OPT)
BaseModel.add_to_registry(OPTLora.config_name, OPTLora)
BaseModel.add_to_registry(OPTInt8.config_name, OPTInt8)
BaseModel.add_to_registry(OPTLoraInt8.config_name, OPTLoraInt8)
BaseModel.add_to_registry(Cerebras.config_name, Cerebras)
BaseModel.add_to_registry(CerebrasLora.config_name, CerebrasLora)
BaseModel.add_to_registry(CerebrasInt8.config_name, CerebrasInt8)
BaseModel.add_to_registry(CerebrasLoraInt8.config_name, CerebrasLoraInt8)
BaseModel.add_to_registry(Bloom.config_name, Bloom)
BaseModel.add_to_registry(BloomLora.config_name, BloomLora)
BaseModel.add_to_registry(BloomInt8.config_name, BloomInt8)
BaseModel.add_to_registry(BloomLoraInt8.config_name, BloomLoraInt8)
BaseModel.add_to_registry(StableDiffusion.config_name, StableDiffusion)
BaseModel.add_to_registry(GenericModel.config_name, GenericModel)
BaseModel.add_to_registry(GenericLoraModel.config_name, GenericLoraModel)
BaseModel.add_to_registry(GenericInt8Model.config_name, GenericInt8Model)
BaseModel.add_to_registry(GenericLoraInt8Model.config_name, GenericLoraInt8Model)
BaseModel.add_to_registry(Falcon.config_name, Falcon)
BaseModel.add_to_registry(FalconLora.config_name, FalconLora)
BaseModel.add_to_registry(FalconInt8.config_name, FalconInt8)
BaseModel.add_to_registry(FalconLoraInt8.config_name, FalconLoraInt8)
