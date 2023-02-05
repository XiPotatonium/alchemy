from copy import deepcopy
from dataclasses import dataclass
from math import ceil
from queue import Queue
import time
from typing import Any, Dict, Iterable, Iterator, List, MutableMapping, Optional, Union
from pathlib import Path
import multiprocessing as mp
import torch

from .util.alloc_gpu import alloc_cuda
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
    with AlchemyRunner.from_registry(cfg["runner"], cfg, device_info, **kwargs) as runner:
        try:
            runner.run()
        except Exception as e:
            sym_tbl().exception = e     # Some plugins may need this value for debugging
            raise e


def _run_task_wrapper(q: Queue, cfg: MutableMapping, device_info: Dict[str, Any], **kwargs):
    try:
        result = run_task(cfg, device_info, **kwargs)
    except Exception as e:
        result = RunResult(
            record_dir=sym_tbl().record_dir,
            cfg=sym_tbl().cfg,
            ret=sym_tbl().ret,
            exception=e,
        )
        raise e
    finally:
        # 无论如何都要put，不然主线程会卡住
        q.put(result)


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
    """注意在多进程（len(cfgs) != 1）的情况下，返回的顺序和cfgs并不一定对应

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
        List[RunResult]: _description_
    """
    ret = []

    if len(cfgs) == 1 and not force_mp:
        device_info = next(alloc_cuda([[] if device is None or len(device) == 0 else device]))
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
            # 发现了一件很奇怪的事情，linux在fork模式下可以用mp.Queue，但是在spawn模式下不行
            # 似乎子进程的queue.put无法正常进行，而且现象很奇怪，像是子进程的put没有报错直接退出了
            # Windows的spawn倒是没有问题，
            # 这里用mgr来托管Queue了，这样搞好像可以运行
            ret_q = mgr.Queue()
            subprocess = []
            ctx = mp.get_context('spawn')
            # alloc device and start training
            for cfg, device_info in zip(
                cfgs, RepeatIter(alloc_cuda(
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
