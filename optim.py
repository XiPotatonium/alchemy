# __future__.annotations will become the default in Python 3.11
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch.optim import Optimizer
from loguru import logger

from .util.sym import sym_tbl
from .registry import Registrable
from .util import filter_optional_cfg


class AlchemyOptimizer(ABC, Registrable):
    @classmethod
    def from_registry(cls, ty: str) -> AlchemyOptimizer:
        optim_cls = cls.resolve_registered_module(ty)
        optimizer = optim_cls()
        return optimizer

    def __init__(self):
        super().__init__()
        self._optimizer: Optimizer

    @property
    def cfg(self) -> Dict[str, Any]:
        return sym_tbl().cfg["optim"]

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Reset optimizer instance."""
        if not isinstance(optimizer, Optimizer):
            logger.warning(
                "Use optimizer ({}) which is not an instance of {}".format(
                    optimizer.__class__.__name__,
                    Optimizer.__name__,
                )
            )
        self._optimizer = optimizer

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @abstractmethod
    def reset(self, **kwargs):
        raise NotImplementedError()

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                if torch.is_tensor(c):
                    c = c.to(p.grad.device)
                p.grad.data.mul_(c)

    def get_lr(self) -> List[float]:
        return [group['lr'] for group in self.param_groups]

    def step(self):
        """alchemy will not call it automatically, you should call it in your task
        """
        self.optimizer.step()


@AlchemyOptimizer.register()
class AdamW(AlchemyOptimizer):
    def __init__(self):
        super(AdamW, self).__init__()
        self.reset()

    def reset(self, **kwargs):
        from torch.optim import AdamW as _TchAdamW

        # 如果kwargs里面没有，那么从self.cfg里面找，如果还是没有，则为默认值
        cfg = dict(self.cfg)
        cfg.update(kwargs)

        self.max_grad_norm = cfg["max_grad_norm"]

        self.optimizer = _TchAdamW(
            sym_tbl().model.optim_params(**cfg),
            **filter_optional_cfg(
                cfg=cfg,
                optional_keys={"lr", "weight_decay", "betas", "eps", "amsgrad"},
            ),
        )

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        super().step()


@AlchemyOptimizer.register()
class HFAdamW(AlchemyOptimizer):
    def __init__(self):
        super(HFAdamW, self).__init__()
        self.reset()

    def reset(self, **kwargs):
        from transformers.optimization import AdamW as _HFAdamW

        # 如果kwargs里面没有，那么从self.cfg里面找，如果还是没有，则为默认值
        cfg = dict(self.cfg)
        cfg.update(kwargs)

        self.max_grad_norm = cfg["max_grad_norm"]

        self.optimizer = _HFAdamW(
            sym_tbl().model.optim_params(**cfg),
            **filter_optional_cfg(
                cfg=cfg,
                optional_keys={"lr", "weight_decay", "betas", "eps", "correct_bias"},
            ),
            no_deprecation_warning=True,
        )

    def step(self):
        torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        super().step()
