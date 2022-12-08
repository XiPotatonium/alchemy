from typing import Any, Dict, MutableMapping
from loguru import logger

from . import (
    SchedPipeline, BeginEpochPipeline, BeginStepPipeline, EndEpochPipeline, EndStepPipeline
)
from ..util.sym import sym_tbl
from ..runner import evaluate


@SchedPipeline.register("SetRequiresGradBSPipeline")
class SetRequiresGradBSPipeline(BeginStepPipeline):
    def __init__(self, step: int, requires_grad: bool, mode: str, **kwargs) -> None:
        super().__init__()
        self.step = step
        self.requires_grad = requires_grad
        self.mode = mode
        self.kwargs = kwargs

    def __call__(self, batch: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        if sym_tbl().train_sched.cur_step == self.step:
            logger.info(
                "Set trf encoder requires_grad to {} at step {}".format(
                    self.requires_grad, sym_tbl().train_sched.cur_step
                )
            )
            sym_tbl().model.set_requires_grad(requires_grad=self.requires_grad, mode=self.mode, **self.kwargs)
        return kwargs


@SchedPipeline.register("InitEvalBEPipeline")
class InitEvalBEPipeline(BeginEpochPipeline):
    def __init__(self, split: str = "dev", needs_loss: bool = True, **kwargs) -> None:
        super().__init__()
        self.split = split
        self.needs_loss = needs_loss

    def __call__(self, **kwargs) -> Dict[str, Any]:
        cur_epoch = sym_tbl().train_sched.cur_epoch
        if cur_epoch == 0:
            logger.info(
                "Init eval at epoch {} step {}".format(cur_epoch, sym_tbl().train_sched.cur_step)
            )
            evaluate(self.split, needs_loss=self.needs_loss)
        return kwargs


@SchedPipeline.register("EndEvalEEPipeline")
class EndEvalEEPipeline(EndEpochPipeline):
    def __init__(self, split: str = "dev", needs_loss: bool = True, **kwargs) -> None:
        super().__init__()
        self.split = split
        self.needs_loss = needs_loss

    def __call__(self, **kwargs) -> Dict[str, Any]:
        cur_epoch = sym_tbl().train_sched.cur_epoch
        # NOTE: 如果和EvalStep和EvalEpoch共用的话可能会导致最后一次eval是重复的
        if cur_epoch + 1 == sym_tbl().train_sched.cfg["epochs"]:
            logger.info(
                "End eval at epoch {} step {}".format(cur_epoch, sym_tbl().train_sched.cur_step)
            )
            evaluate(self.split, needs_loss=self.needs_loss)
        return kwargs


@SchedPipeline.register("EvalStepESPipeline")
class EvalStepESPipeline(EndStepPipeline):
    def __init__(self, period: int, split: str = "dev", needs_loss: bool = True, **kwargs) -> None:
        super().__init__()
        self.period = period
        self.split = split
        self.needs_loss = needs_loss

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        cur_step = sym_tbl().train_sched.cur_step
        if (cur_step + 1) % self.period == 0:
            logger.info(
                "EvalStep at epoch {} step {}".format(sym_tbl().train_sched.cur_epoch, sym_tbl().train_sched.cur_step)
            )
            evaluate(self.split, needs_loss=self.needs_loss)
        return kwargs


@SchedPipeline.register("EvalEpochEEPipeline")
class EvalEpochEEPipeline(EndEpochPipeline):
    def __init__(self, period: int, split: str = "dev", needs_loss: bool = True, **kwargs) -> None:
        super().__init__()
        self.period = period
        self.split = split
        self.needs_loss = needs_loss

    def __call__(self, **kwargs) -> Dict[str, Any]:
        cur_epoch = sym_tbl().train_sched.cur_epoch
        if (cur_epoch + 1) % self.period == 0:
            logger.info(
                "EvalEpoch at epoch {} step {}".format(
                    sym_tbl().train_sched.cur_epoch,
                    sym_tbl().train_sched.cur_step
                )
            )
            evaluate(self.split, needs_loss=self.needs_loss)
        return kwargs
