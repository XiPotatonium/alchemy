# __future__.annotations will become the default in Python 3.11
from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Optional
from abc import ABC, abstractmethod
import random
from datetime import datetime

import numpy as np

from rich.progress import Progress

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .registry import Registrable
from .util.extention.rich import full_columns, no_total_columns
from .util.sym import sym_tbl
from .task import AlchemyTask
from .model import AlchemyModel
from .optim import AlchemyOptimizer
from .scheduler import AlchemyTrainScheduler
from .plugins import AlchemyPlugin


class AlchemyRunner(Registrable, ABC):
    """Runner is a wrapper for run logic

    Args:
        Registrable (_type_): _description_
        ABC (_type_): _description_

    Returns:
        _type_: _description_
    """
    @classmethod
    def from_registry(cls, ty: str, cfg: MutableMapping, device_info: Dict[str, Any], **kwargs) -> AlchemyRunner:
        runner_cls = cls.resolve_registered_module(ty)
        runner = runner_cls(cfg, device_info, **kwargs)
        return runner

    def __init__(
        self,
        cfg: MutableMapping[str, Any],
        device_info: Dict[str, Any],
        **kwargs,
    ):
        # DO NOT place other fields in runner, place them in sym_tbl
        sym_tbl().cfg = cfg
        sym_tbl().device_info = device_info
        sym_tbl().device = torch.device(device_info["device"])
        sym_tbl().ctime = datetime.now()

        for key, val in kwargs.items():
            sym_tbl().set_global(key, val)

        # plugins
        for p_cfg in sym_tbl().cfg.get("plugins", []):
            sym_tbl().plugins.append(AlchemyPlugin.from_registry(p_cfg["type"], **p_cfg))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in sym_tbl().plugins:
            p.exit()

    @abstractmethod
    def run(self):
        pass


@AlchemyRunner.register("Trainer")
class AlchemyTrainer(AlchemyRunner):
    def __init__(self, cfg: MutableMapping[str, Any], device_info: Dict[str, Any], **kwargs):
        super().__init__(cfg, device_info, **kwargs)

    def run(self):
        sym_tbl().task = AlchemyTask.from_registry(sym_tbl().cfg["task"]["type"])
        sym_tbl().model = AlchemyModel.from_registry(sym_tbl().cfg["model"]["type"])
        sym_tbl().model.to(device=sym_tbl().device)     # occupy GPU as soon as possible

        # 'train' and 'dev' are required
        for split, dataset_cfg in sym_tbl().cfg["task"]["datasets"].items():
            sym_tbl().task.load_dataset(split=split, **dataset_cfg)

        sym_tbl().optim = AlchemyOptimizer.from_registry(sym_tbl().cfg["optim"]["type"])
        sym_tbl().train_sched = AlchemyTrainScheduler.from_registry(sym_tbl().cfg["sched"]["type"])

        for epoch in sym_tbl().train_sched:
            dset, dset_kw = sym_tbl().task.dataset('train')
            itr = get_dataloader(
                dset,
                **dset_kw,
            )
            total = len(itr) if not isinstance(dset, IterableDataset) else float('inf')
            columns = full_columns() if not isinstance(dset, IterableDataset) else no_total_columns()

            with Progress(*columns, console=sym_tbl().console, disable=sym_tbl().console is None) as pbar:
                sym_tbl().set_global("pbar", pbar)          # 有时候可能会有多个progress任务（例如dev），为了显示美观，让下面的任务可以复用pbar
                tid = pbar.add_task("Epoch{}".format(epoch), total=total)
                for batch in itr:
                    step, _ = sym_tbl().train_sched.begin_train_step(batch)
                    sym_tbl().model.train()
                    train_log, outputs = sym_tbl().task.step(batch)
                    sym_tbl().train_sched.end_train_step(outputs)
                    pbar.update(
                        tid,
                        advance=1,
                        description="Epoch{},{}".format(epoch, train_log) if len(train_log) != 0 else "Epoch{}".format(epoch)
                    )
                sym_tbl().pop_global("pbar")

            sym_tbl().train_sched.end_train_epoch()


@AlchemyRunner.register("Tester")
class ALchemyTester(AlchemyRunner):
    def run(self):
        sym_tbl().task = AlchemyTask.from_registry(sym_tbl().cfg["task"]["type"])
        sym_tbl().model = AlchemyModel.from_registry(sym_tbl().cfg["model"]["type"])
        sym_tbl().model.to(sym_tbl().device)     # occupy GPU as soon as possible

        # 'test' is required
        for split, dataset_cfg in sym_tbl().cfg["task"]["datasets"].items():
            sym_tbl().task.load_dataset(split=split, **dataset_cfg)

        evaluate(split="test", needs_loss=False)


@AlchemyRunner.register("InferenceRunner")
class AlchemyInferenceRunner(AlchemyRunner):
    def run(self):
        sym_tbl().task = AlchemyTask.from_registry(sym_tbl().cfg["task"]["type"])
        sym_tbl().model = AlchemyModel.from_registry(sym_tbl().cfg["model"]["type"])
        sym_tbl().model.to(sym_tbl().device)

        for split, dataset_cfg in sym_tbl().cfg["task"]["datasets"].items():
            sym_tbl().task.load_dataset(split=split, **dataset_cfg)

        dset, dset_kw = sym_tbl().task.dataset("inference")
        itr = get_dataloader(
            dset,
            **dset_kw,
        )
        total = len(itr) if not isinstance(dset, IterableDataset) else float('inf')
        columns = full_columns() if not isinstance(dset, IterableDataset) else no_total_columns()

        with Progress(*columns, console=sym_tbl().console, disable=sym_tbl().console is None) as pbar:
            tid = pbar.add_task("Inferencing", total=total)
            sym_tbl().model.eval()
            with torch.no_grad():
                for batch in itr:
                    sym_tbl().task.eval_step(
                        batch,
                        needs_loss=False,
                    )
                    pbar.advance(tid)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(
    dataset: Dataset,
    seed=0,
    **kwargs,
):
    """AlchemyRunner will iterate dataset with this method

    Args:
        dataset (Dataset):
        seed (int, optional): 用于generator. Defaults to 1.
        num_workers (int, optional): _description_. Defaults to 0.
    """

    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        worker_init_fn=seed_worker,
        generator=g,
        **kwargs,
    )


def evaluate(
    split: str = "dev",
    needs_loss: bool = True,
    **kwargs,
):
    pbar: Progress = sym_tbl().try_get_global("pbar")

    dset, dset_kw = sym_tbl().task.dataset(split)
    itr = get_dataloader(
        dset,
        **dset_kw,
    )

    sym_tbl().task.begin_eval(split, **kwargs)
    sym_tbl().model.eval()
    new_pbar = pbar is None
    if new_pbar:
        columns = full_columns() if not isinstance(dset, IterableDataset) else no_total_columns()
        pbar = Progress(*columns, console=sym_tbl().console, disable=sym_tbl().console is None)
        pbar.start()
    total = len(itr) if not isinstance(dset, IterableDataset) else float('inf')
    tid = pbar.add_task("Eval", total=total)

    with torch.no_grad():
        for batch in itr:
            eval_log, output = sym_tbl().task.eval_step(batch, needs_loss=needs_loss)
            pbar.update(
                tid,
                advance=1,
                description="Eval,{}".format(eval_log) if len(eval_log) != 0 else "Eval",
            )
    sym_tbl().task.end_eval(split, **kwargs)
    if new_pbar:
        pbar.stop()
