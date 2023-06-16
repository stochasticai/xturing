import json
from pathlib import Path
from typing import Iterable, List, Optional, Union

import torch
from pytorch_lightning.loggers import Logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from xturing.config import DEFAULT_DEVICE, assert_not_cpu_int8
from xturing.config.config_data_classes import FinetuningConfig, GenerationConfig
from xturing.config.read_config import load_config
from xturing.datasets.instruction_dataset import InstructionDataset
from xturing.datasets.text_dataset import TextDataset
from xturing.engines.base import BaseEngine
from xturing.models import BaseModel
from xturing.preprocessors.base import BasePreprocessor
from xturing.trainers.base import BaseTrainer
from xturing.trainers.lightning_trainer import LightningTrainer
from xturing.utils.logging import configure_logger
from xturing.utils.utils import _filter_args

logger = configure_logger(__name__)


class CausalModel(BaseModel):
    def __init__(
        self,
        engine: str,
        weights_path: Optional[str] = None,
        model_name: Optional[str] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        arguments = dict(
            weights_path=weights_path,
            model_name=model_name,
            target_modules=target_modules,
            **kwargs,
        )

        self.engine = BaseEngine.create(
            engine,
            **_filter_args(arguments),
        )

        self.model_name = engine.replace("_engine", "")

        # Finetuning config
        self.finetuning_args = load_config(
            model_name=self.model_name,
            config_path=Path(__file__).parent.parent
            / "config"
            / "finetuning_config.yaml",
            data_class=FinetuningConfig,
        )

        # Generation config

        self.generation_args = load_config(
            model_name=engine.replace("_engine", ""),
            config_path=Path(__file__).parent.parent
            / "config"
            / "generation_config.yaml",
            data_class=GenerationConfig,
        )

        logger.debug(f"Finetuning parameters: {self.finetuning_args}")
        logger.debug(f"Generation parameters: {self.generation_args}")

    def finetuning_config(self):
        return self.finetuning_args

    def generation_config(self):
        return self.generation_args

    def _make_collate_fn(self, dataset: Union[TextDataset, InstructionDataset]):
        return BasePreprocessor.create(
            dataset.config_name,
            self.engine.tokenizer,
            int(self.finetuning_args.max_length),
            dataset.meta,
        )

    def _make_trainer(
        self,
        dataset: Union[TextDataset, InstructionDataset],
        logger: Union[Logger, Iterable[Logger], bool] = True,
    ):
        return BaseTrainer.create(
            LightningTrainer.config_name,
            self.engine,
            dataset,
            self._make_collate_fn(dataset),
            int(self.finetuning_args.num_train_epochs),
            int(self.finetuning_args.batch_size),
            float(self.finetuning_args.learning_rate),
            self.finetuning_args.optimizer_name,
            logger=logger,
        )

    def finetune(
        self,
        dataset: Union[TextDataset, InstructionDataset],
        logger: Union[Logger, Iterable[Logger], bool] = True,
    ):
        assert dataset.config_name in [
            "text_dataset",
            "instruction_dataset",
        ], "Please make sure the dataset_type is text_dataset or instruction_dataset"
        trainer = self._make_trainer(dataset, logger)
        trainer.fit()

    def evaluate(self, dataset: Union[TextDataset, InstructionDataset]):
        pass

    def _generate_from_iterable(
        self, data_iterator: Iterable, do_tokenization=False, show_tqdm_bar=True
    ):
        outputs = []

        if show_tqdm_bar:
            enumeration = enumerate(tqdm(data_iterator))
        else:
            enumeration = enumerate(data_iterator)

        for i, batch in enumeration:
            if do_tokenization:
                inputs = self.engine.tokenizer(batch, return_tensors="pt")
                input_ids = inputs.input_ids.to(DEFAULT_DEVICE)
            else:
                input_ids = batch["input_ids"].to(DEFAULT_DEVICE)
            with torch.no_grad():
                with torch.autocast("cuda"):
                    len_input = input_ids.shape[1]
                    output = self.engine.model.generate(
                        input_ids=input_ids, **self.generation_args.dict()
                    )

            output = self.engine.tokenizer.decode(
                output[0][len_input:], skip_special_tokens=True
            )
            outputs.append(output)

        return outputs

    def generate(
        self,
        *,
        texts: Optional[Union[List[str], str]] = None,
        dataset: Optional[Union[TextDataset, InstructionDataset]] = None,
    ):
        self.engine.model.eval()
        self.engine.model = self.engine.model.to(DEFAULT_DEVICE)

        outputs = []

        if texts is not None:
            flattened_texts = [texts] if isinstance(texts, str) else texts

            outputs.extend(
                self._generate_from_iterable(
                    flattened_texts, do_tokenization=True, show_tqdm_bar=False
                )
            )

        if dataset is not None:
            collate_fn = self._make_collate_fn(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn,
            )

            outputs.extend(
                self._generate_from_iterable(dataloader, do_tokenization=False)
            )

        if texts is None and dataset is None:
            assert False, "Make sure texts or dataset is not None"

        if isinstance(texts, str) and dataset is None:
            return outputs[0]

        return outputs

    def _save_config(self, path: Union[str, Path]):
        xturing_config_path = Path(path) / "xturing.json"
        xturing_config = {
            "model_name": self.model_name,
            "finetuning_config": self.finetuning_args.dict(),
            "generation_config": self.generation_args.dict(),
        }

        with open(str(xturing_config_path), "w", encoding="utf-8") as f:
            json.dump(xturing_config, f, ensure_ascii=False, indent=4)

    def save(self, path: Union[str, Path]):
        path = Path(path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

        self.engine.save(path)
        self._save_config(path=path)


class CausalInt8Model(CausalModel):
    def __init__(
        self,
        engine: str,
        weights_path: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        assert_not_cpu_int8()
        super().__init__(
            engine, weights_path=weights_path, model_name=model_name, **kwargs
        )


class CausalLoraModel(CausalModel):
    def __init__(
        self,
        engine: str,
        weights_path: Optional[str] = None,
        model_name: Optional[str] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            engine,
            weights_path=weights_path,
            model_name=model_name,
            target_modules=target_modules,
            **kwargs,
        )

    def _make_trainer(
        self,
        dataset: Union[TextDataset, InstructionDataset],
        logger: Union[Logger, Iterable[Logger], bool] = True,
    ):
        return BaseTrainer.create(
            LightningTrainer.config_name,
            self.engine,
            dataset,
            self._make_collate_fn(dataset),
            int(self.finetuning_args.num_train_epochs),
            int(self.finetuning_args.batch_size),
            float(self.finetuning_args.learning_rate),
            self.finetuning_args.optimizer_name,
            True,
            True,
            logger=logger,
        )


class CausalLoraInt8Model(CausalLoraModel):
    def __init__(
        self,
        engine: str,
        weights_path: Optional[str] = None,
        model_name: Optional[str] = None,
        target_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        assert_not_cpu_int8()
        super().__init__(
            engine,
            weights_path=weights_path,
            model_name=model_name,
            target_modules=target_modules,
            **kwargs,
        )
