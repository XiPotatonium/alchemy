from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from queue import Queue
import time
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Union
from pathlib import Path
import multiprocessing as mp
import torch

from .util.device import alloc
from .util.sym import sym_tbl, new_scope
from .model import AlchemyModel
from .task import AlchemyTask
from .runner import AlchemyRunner
from .sched import AlchemyTrainScheduler
from .optim import AlchemyOptimizer


@dataclass
class RunResult:
    record_dir: Optional[Path]
    cfg: MutableMapping
    ret: Any
    exception: Optional[Exception]


class RepeatIter(Iterator):
    def __init__(self, data: Union[Iterator, Iterable], repeat: int = 1) -> None:
        super().__init__()
        if isinstance(data, Iterator):
            self.data = data
        elif isinstance(data, Iterable):
            self.data = iter(data)
        else:
            raise ValueError("Unsupported data type {}".format(type(data)))
        self.repeat = repeat
        self.cur = None
        self.cur_repeat = 0

    def __next__(self):
        if self.cur_repeat >= self.repeat or self.cur is None:
            self.cur_repeat = 1
            self.cur = next(self.data)
        else:
            self.cur_repeat += 1
        return self.cur


def prepare_cfg(cfg: Union[Path, MutableMapping]) -> MutableMapping:
    import tomlkit
    if isinstance(cfg, MutableMapping):
        cfg = deepcopy(cfg)
    elif isinstance(cfg, Path):
        with cfg.open('r', encoding="utf8") as f:
            cfg = tomlkit.load(f)
    else:
        raise ValueError(
            f"Expect cfg to be {Path.__name__} or {MutableMapping.__name__} "
            f"but found {cfg.__class__.__name__}"
        )
    return cfg


def run_task(cfg: MutableMapping, device_info: Dict[str, Any], **kwargs) -> RunResult:
    try:
        with AlchemyRunner.from_registry(cfg["runner"], cfg, device_info, **kwargs) as runner:
            try:
                runner.run()
            except Exception as e:
                # Catch exception in running
                # Exceptions in running can be used in plugins for debugging
                sym_tbl().exception = e
    except Exception as e:
        # Exceptions in runner initialization cannot be used in plugins
        # But it should be captured here
        # This method will not throw
        sym_tbl().exception = e
    return RunResult(
        record_dir=sym_tbl().record_dir,
        cfg=sym_tbl().cfg,
        ret=sym_tbl().ret,
        exception=e,
    )


def _run_task_wrapper(q: Queue, cfg: MutableMapping, device_info: Dict[str, Any], **kwargs):
    q.put(run_task(cfg, device_info, **kwargs))


def run(
    cfgs: List[MutableMapping],
    device: Optional[List[int]] = None,
    user_dir: str = "src",
    desc: str = "",
    debug: bool = False,
    no_file: bool = False,
    force_mp: bool = False,
    task_per_device: int = 1,
) -> List[RunResult]:
    """

    Args:
        cfgs (List[MutableMapping]): _description_
        device (Optional[List[int]], optional): _description_. Defaults to None.
        user_dir (str, optional): _description_. Defaults to "src".
        desc (str, optional): _description_. Defaults to "".
        debug (bool, optional): _description_. Defaults to False.
        no_file (bool, optional): _description_. Defaults to False.
        force_mp (bool, optional): _description_. Defaults to False.
        task_per_device (int, optional): _description_. Default to 1.

    Returns:
        List[RunResult]: The order of the return results is not necessarily same as the cfgs
    """
    ret = []

    if len(cfgs) == 1 and not force_mp:
        device_info = next(alloc([[] if device is None or len(device) == 0 else device]))
        ret.append(run_task(
            cfg=cfgs[0],
            device_info=device_info,
            user_dir=user_dir,
            desc=desc,
            debug=debug,
            no_file=no_file,
        ))
        # 释放模型、数据集等内容，避免占用
        sym_tbl().reset()
        torch.cuda.empty_cache()
    else:
        with mp.Manager() as mgr:
            ret_q = mgr.Queue()
            subprocess = []
            ctx = mp.get_context('spawn')
            # alloc device and start training
            for cfg, device_info in zip(
                cfgs, RepeatIter(alloc(
                    [[] if device is None or len(device) == 0 else device for _ in range(ceil(len(cfgs) / task_per_device))],
                ), repeat=task_per_device)      # alloc n tasks per free gpu
            ):
                p = ctx.Process(
                    target=_run_task_wrapper,
                    args=(ret_q, cfg, device_info),
                    kwargs={
                        "user_dir": user_dir,
                        "desc": desc,
                        "debug": debug,
                        "no_file": no_file,
                    }
                )
                subprocess.append(p)
                p.start()
                time.sleep(1)

            for p in subprocess:
                p.join()
                ret.append(ret_q.get())
    return ret
