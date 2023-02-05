# __future__.annotations will become the default in Python 3.11
from __future__ import annotations

from typing import Any, Dict, Iterator, List, MutableMapping
import warnings
from torch.optim.lr_scheduler import _LRScheduler

import torch

from .pipeline import (
    BeginEpochPipeline,
    BeginStepPipeline,
    SchedPipeline,
    EndEpochPipeline,
    EndStepPipeline
)

from .registry import Registrable
from .util import filter_optional_cfg
from .util.sym import sym_tbl


class AlchemyTrainScheduler(Registrable):
    @classmethod
    def from_registry(cls, ty: str) -> AlchemyTrainScheduler:
        sched_cls = cls.resolve_registered_module(ty)
        train_sched = sched_cls()
        return train_sched

    def __init__(self):
        super(AlchemyTrainScheduler, self).__init__()
        self.lr_sched = None

        self.cur_epoch = -1
        self.cur_step = -1
        self.cur_step_this_epoch = -1

        self.begin_epoch_pipes: List[BeginEpochPipeline] = []
        self.begin_step_pipes: List[BeginStepPipeline] = []
        self.end_epoch_pipes: List[EndEpochPipeline] = []
        self.end_step_pipes: List[EndStepPipeline] = []

        for callback_cfg in self.cfg.get("pipes", []):
            callback_cfg: Dict[str, Any] = callback_cfg
            ty = callback_cfg["type"]
            callback = SchedPipeline.from_registry(ty, **callback_cfg)
            # 注意一个callback如果继承了多种Callback那么可以在多个时机被触发，但其实不推荐这么做
            if isinstance(callback, BeginEpochPipeline):
                self.begin_epoch_pipes.append(callback)
            if isinstance(callback, EndEpochPipeline):
                self.end_epoch_pipes.append(callback)
            if isinstance(callback, BeginStepPipeline):
                self.begin_step_pipes.append(callback)
            if isinstance(callback, EndStepPipeline):
                self.end_step_pipes.append(callback)

    @property
    def cfg(self) -> Dict[str, Any]:
        return sym_tbl().cfg["sched"]

    def __iter__(self) -> Iterator:
        for _ in range(self.cfg["epochs"]):
            yield self._begin_train_epoch()

    def reset_lr_sched(self, **kwargs):
        """reset lr_scheduler, DO NOT reset entire TrainScheduler because pipes will be re-intialized
        """
        pass

    def _begin_train_epoch(self, **kwargs) -> int:
        self.cur_step_this_epoch = -1
        self.cur_epoch += 1
        for e in self.begin_epoch_pipes:
            kwargs = e(**kwargs)
        return self.cur_epoch

    def end_train_epoch(self, **kwargs):
        for e in self.end_epoch_pipes:
            kwargs = e(**kwargs)

    def begin_train_step(self, batch: MutableMapping[str, Any], **kwargs):
        self.cur_step += 1
        self.cur_step_this_epoch += 1
        for e in self.begin_step_pipes:
            kwargs = e(batch, **kwargs)
        return self.cur_step, self.cur_step_this_epoch

    def end_train_step(self, outputs: MutableMapping[str, Any], **kwargs):
        for e in self.end_step_pipes:
            kwargs = e(outputs, **kwargs)

    def step_lr(self):
        """alchemy will not call it automatically, you should call it in your task
        """
        if self.lr_sched is not None:
            self.lr_sched.step()


class NoamScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model_size: int,
        num_warmup_steps: int = 0,
        factor: float = 1.0,
        last_epoch=-1,
        verbose=False
    ):
        self._num_warmup_steps = num_warmup_steps
        self._factor = factor
        self._model_size = model_size

        # step() and get_lr will be called in initialization to set lr to params
        # therefore _factor should be set before super().__init__()
        super().__init__(optimizer, last_epoch, verbose)

        if last_epoch != -1:
            raise NotImplementedError()

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        lr = self.lr_formula(self._factor, self._model_size, self._num_warmup_steps, self._step_count)
        return [lr for _ in self.optimizer.param_groups]

    @staticmethod
    def lr_formula(factor: float, model_size: float, num_warmup_steps: int, step: int):
        if num_warmup_steps == 0:
            return factor * model_size ** (-0.5) * step ** (-0.5)
        return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * num_warmup_steps ** (-1.5))
        )


@AlchemyTrainScheduler.register()
class LineWarmup(AlchemyTrainScheduler):
    def __init__(self):
        super(LineWarmup, self).__init__()
        self.reset_lr_sched()

    def reset_lr_sched(self, **kwargs):
        import transformers
        # NOTE: 如果dataset是AlchemyIterDset那么len可能不准确

        dataset, dataset_cfg = sym_tbl().task.dataset("train")
        # 如果kwargs里面没有，那么从self.cfg里面找，如果还是没有，则为默认值
        cfg = dict(self.cfg)
        cfg.update(kwargs)

        # 如果没写明多少个step（大部分情况下是的），就自动计算总共有多少个step，
        # 但是这种方法在AlchemyItrDset上可能不行因为AlchemyItrDset不一定有__len__
        self.lr_sched = transformers.get_linear_schedule_with_warmup(
            sym_tbl().optim.optimizer,
            num_warmup_steps=cfg.get("num_warmup_steps", 0),
            num_training_steps=cfg.get("num_training_steps", len(dataset) * self.cfg["epochs"] - self.cur_step),
        )


@AlchemyTrainScheduler.register()
class Noam(AlchemyTrainScheduler):
    def __init__(self):
        super(Noam, self).__init__()
        self.reset_lr_sched()

    def reset_lr_sched(self, **kwargs):
        # https://docs.allennlp.org/main/api/training/learning_rate_schedulers/noam/#noamlr
        # 如果kwargs里面没有，那么从self.cfg里面找，如果还是没有，则为默认值
        cfg = dict(self.cfg)
        cfg.update(kwargs)

        self.lr_sched = NoamScheduler(
            sym_tbl().optim.optimizer,
            last_epoch=-1,
            verbose=False,
            model_size=sym_tbl().model.config.hidden_size,
            **filter_optional_cfg(cfg, {"num_warmup_steps", "factor"})
        )
