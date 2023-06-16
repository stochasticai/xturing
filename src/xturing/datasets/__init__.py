from xturing.datasets.base import BaseDataset
from xturing.datasets.instruction_dataset import (
    InstructionDataset,
    InstructionDatasetMeta,
)
from xturing.datasets.text2image_dataset import Text2ImageDataset
from xturing.datasets.text_dataset import TextDataset, TextDatasetMeta

BaseDataset.add_to_registry(TextDataset.config_name, TextDataset)
BaseDataset.add_to_registry(InstructionDataset.config_name, InstructionDataset)
BaseDataset.add_to_registry(Text2ImageDataset.config_name, Text2ImageDataset)
