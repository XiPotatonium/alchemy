# __future__.annotations will become the default in Python 3.11
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, MutableMapping, Tuple, Union
from loguru import logger
from torch.utils.data import Dataset

from .pipeline import DataPipeline, EvalPipeline, OutputPipeline
from .registry import Registrable
from .util.sym import sym_tbl


class AlchemyTask(ABC, Registrable):
    @classmethod
    def from_registry(cls, ty: str) -> AlchemyTask:
        task_cls = cls.resolve_registered_module(ty)
        task = task_cls()
        return task

    def __init__(self):
        self._datasets: Dict[str, Tuple[Dataset, Dict[str, Any]]] = dict()

        self.outputpipes: List[OutputPipeline] = []
        self.evalpipes: List[EvalPipeline] = []

    @property
    def cfg(self) -> Dict[str, Any]:
        return sym_tbl().cfg["task"]

    def max_positions(self):
        """Return the max input length allowed by the task."""
        return None

    def dataset(self, split: str) -> Tuple[Dataset, Dict[str, Any]]:
        """_summary_

        Args:
            split (str): _description_

        Raises:
            KeyError: _description_

        Returns:
            Tuple[Dataset, Dict[str, Any]]: dataset and args. The args will be passed to DataLoader
        """
        if split not in self._datasets:
            raise KeyError("Dataset not loaded: " + split)
        return self._datasets[split]

    def load_dataset(
        self,
        split: str,
        **kwargs,
    ):
        """
        You may have to modify some kwargs such as collate_fn
        """
        logger.info(f"Load dataset {split}")

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = None
        pipes: List[Dict[str, Any]] = kwargs.pop("pipes")

        for i, p_cfg in enumerate(pipes):
            if i == 0:
                datapipe = DataPipeline.from_registry(p_cfg["type"], **p_cfg)
            else:
                datapipe = DataPipeline.from_registry(p_cfg["type"], datapipe, **p_cfg)

        self._datasets[split] = (datapipe, kwargs)

    def step(
        self,
        batch: MutableMapping,
        **kwargs
    ) -> Tuple[str, Union[Dict[str, Any], List]]:
        outputs = sym_tbl().model.forward(
            batch,
            needs_loss=True,
            requires_grad=True,
        )

        return "l={:.4g}".format(outputs["loss"]), outputs

    def eval_step(
        self,
        batch: MutableMapping,
        needs_loss: bool,
        **kwargs
    ) -> Tuple[str, Union[Dict[str, Any], List]]:
        outputs = sym_tbl().model.forward(
            batch,
            needs_loss=needs_loss,
            requires_grad=False,
        )

        if needs_loss:
            loss = outputs["loss"]
            log = "l={:.4g}".format(loss)
        else:
            log = ""

        for p in self.outputpipes:
            outputs = p(outputs, batch)

        return log, outputs

    def begin_eval(self, split: str, **kwargs):
        for pipe in self.evalpipes:
            pipe.begin_eval(split, **kwargs)

    def end_eval(self, split: str, **kwargs):
        for pipe in self.evalpipes:
            kwargs = pipe(split, **kwargs)
