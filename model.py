# __future__.annotations will become the default in Python 3.11
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, MutableMapping, Union, List, Optional
from loguru import logger

from torch import Tensor
from torch.nn import Module, DataParallel
from torch.nn.parallel import DistributedDataParallel

from .registry import Registrable
from .util.sym import sym_tbl


class AlchemyModel(ABC, Registrable):
    @classmethod
    def from_registry(cls, ty: str) -> AlchemyModel:
        model_cls = cls.resolve_registered_module(ty)
        model = model_cls()
        return model

    def __init__(self):
        super().__init__()
        self._model: Union[Module, DistributedDataParallel]       # should be constructed in __init__
        if self.backward_cfg is None:
            self.backward: BackwardHandler = DefaultBackwardHandler()
        else:
            self.backward: BackwardHandler = BackwardHandler.from_registry(
                self.backward_cfg["type"], **self.backward_cfg
            )

    @property
    def model_cfg(self) -> Dict[str, Any]:
        return sym_tbl().cfg["model"]

    @property
    def criterion_cfg(self) -> Optional[Dict[str, Any]]:
        """might be None in tasks such as inference or test-only

        Returns:
            Optional[Dict[str, Any]]: _description_
        """
        return sym_tbl().cfg.get("criterion")

    @property
    def backward_cfg(self) -> Optional[Dict[str, Any]]:
        return sym_tbl().cfg.get("backward")

    @property
    def module(self) -> Module:
        options = (DistributedDataParallel, DataParallel)
        model = self._model
        while isinstance(model, options):
            model = model.module
        return model

    @property
    def model(self) -> Union[Module, DistributedDataParallel]:
        return self._model

    @model.setter
    def model(self, model: Union[Module, DistributedDataParallel]):
        if not isinstance(model, (Module, DistributedDataParallel)):
            logger.warning(
                "Set model ({}) which is not an instance of {} or {}".format(
                    model.__class__.__name__,
                    Module.__name__, DistributedDataParallel.__name__,
                )
            )
        self._model = model

    def train(self, mode: bool = True):
        self.model.train(mode)

    def eval(self):
        self.model.eval()

    def to(self, *args, **kwargs):
        # 偷了个懒，没有写type hint
        return self.model.to(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def set_requires_grad(self, requires_grad: bool, mode: str, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def optim_params(self, **kwargs):
        pass

    @abstractmethod
    def forward(
        self,
        batch: MutableMapping[str, Any],
        needs_loss: bool,
        requires_grad: bool,
        **kwargs
    ) -> MutableMapping[str, Any]:
        """
        推荐在alchemy_forward中处理loss（完成loss的backward而不是将这个工作留给调用者，因为有可能会需要梯度累加）

        Args:
            batch (Dict): batch data
            needs_loss (bool): calc loss or not
            requires_grad (bool): perform backward algorithm (optim.zero_grad(), loss.backward(), optim.step(), sched.step()) or not

        Returns:
            MutableMapping[str, Any]: contains "loss" (if needs_loss)
        """
        pass


class BackwardHandler(Registrable):
    @classmethod
    def from_registry(cls, ty: str, **kwargs):
        handler_cls = cls.resolve_registered_module(ty)
        handler = handler_cls(**kwargs)
        return handler

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def backward(
        self,
        losses: Union[List[Tensor], Tensor],
        weights: Optional[List[Tensor]] = None,
        requires_grad: bool = True,
        names: Optional[List[str]] = None,
    ) -> float:
        """_summary_

        Args:
            losses (Union[List[Tensor], Tensor]): _description_
            weights (Optional[List[Tensor]]): 如果是None，默认权重相同（w=1）
            requires_grad (bool, optional): 是否计算梯度以及优化参数，如果是False那么仅计算loss加权和. Defaults to True.
            names (Optional[List[str]], optional): 每个loss的名字，debug可用，不必要. Defaults to None.

        Raises:
            NotImplementedError: _description_

        Returns:
            float: _description_
        """
        raise NotImplementedError()


@BackwardHandler.register("Default")
class DefaultBackwardHandler(BackwardHandler):
    def __init__(self, period: int = 1, **kwargs) -> None:
        super().__init__()
        self.period = period       # backward period
        if self.period <= 0:
            raise ValueError("Invalid period {}".format(self.period))

    def normal_backward(
        self,
        losses: List[Tensor],
        weights: Optional[List[Tensor]] = None,
        requires_grad: bool = True,
        names: Optional[List[str]] = None,
    ) -> float:
        if weights is None:
            loss = sum(losses)
        else:
            loss = sum(l * w for l, w in zip(losses, weights))

        if requires_grad:
            cur_step: int = sym_tbl().train_sched.cur_step
            if cur_step % self.period == 0:
                sym_tbl().optim.zero_grad()
            if (cur_step + 1) % self.period == 0:
                # 如果当前step是这个period的最后一个step，那么不能再retain_graph了
                loss.backward()
                sym_tbl().optim.step()
                sym_tbl().train_sched.step_lr()
            else:
                loss.backward(retain_graph=True)

        return loss.item()

    def debug_backward(
        self,
        losses: List[Tensor],
        weights: Optional[List[Tensor]] = None,
        requires_grad: bool = True,
        names: Optional[List[str]] = None,
    ) -> float:
        if weights is None:
            loss = sum(losses)
            weights = [1 for _ in range(len(losses))]
        else:
            loss = sum(l * w for l, w in zip(losses, weights))
            assert len(losses) == len(weights)

        if requires_grad:
            if names is None:
                names = ["unnamed{}".format(i) for i in range(len(losses))]
            else:
                assert len(losses) == len(names)

            cur_step: int = sym_tbl().train_sched.cur_step
            if cur_step % self.period == 0:
                sym_tbl().optim.zero_grad()
            if (cur_step + 1) % self.period == 0:
                # 如果当前step是这个period的最后一个step，那么不能再retain_graph了
                for i, (l, w, name) in enumerate(zip(losses, weights, names)):
                    try:
                        if i == len(losses):
                            (l * w).backward()
                        else:
                            (l * w).backward(retain_graph=True)
                    except RuntimeError as e:
                        logger.error("Error bp {}".format(name))
                        raise e
                sym_tbl().optim.step()
                sym_tbl().train_sched.step_lr()
            else:
                loss.backward(retain_graph=True)

        return loss.item()

    def backward(
        self,
        losses: Union[List[Tensor], Tensor],
        weights: Optional[List[Tensor]] = None,
        requires_grad: bool = True,
        names: Optional[List[str]] = None,
    ) -> float:
        losses = [losses] if isinstance(losses, Tensor) else losses
        if sym_tbl().try_get_global("debug", False):
            return self.debug_backward(losses, weights, requires_grad, names)
        else:
            return self.normal_backward(losses, weights, requires_grad, names)
