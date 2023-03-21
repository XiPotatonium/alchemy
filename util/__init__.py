from pathlib import Path
import traceback
from typing import Any, MutableMapping, Optional, Dict, Set
import torch
from torch import Tensor

from loguru import logger


def merge_max_position_constraints(*args: Optional[int]) -> Optional[int]:
    min_max = None
    for constraint in args:
        if constraint is None:
            continue
        elif min_max is None:
            min_max = constraint
        elif constraint < min_max:
            min_max = constraint
    return min_max


def filter_optional_cfg(cfg: Dict[str, Any], optional_keys: Set[str], level: str = "WARNING") -> Dict[str, Any]:
    """一个简单的helper，这里只是为了增加一个logger的输出，防呆设计，做一个提示避免出错

    Args:
        cfg (Dict[str, Any]): _description_
        optional_keys (Set[str]): _description_
        level (str, optional): _description_. Default to be "WARNING"

    Returns:
        Dict[str, Any]: _description_
    """
    ret = {k: v for k, v in cfg.items() if k in optional_keys}
    not_set = optional_keys - set(ret.keys())
    if len(not_set) != 0:
        summary = traceback.StackSummary.extract(
            traceback.walk_stack(None)
        )
        # print(''.join(summary.format()))
        logger.log(
            level,
            "{} not set, will be overriden by default values. (called by {}:{})".format(
                not_set, summary[1].filename, summary[1].lineno
            )
        )
    return ret


def warn_unused_kwargs(kwargs: Dict[str, Any], called_by: Optional[Any] = None, level: str = "WARNING"):
    if len(kwargs) != 0:
        if called_by is None:
            summary = traceback.StackSummary.extract(
                traceback.walk_stack(None)
            )
            called_by = "{}:{}".format(summary[1].filename, summary[1].lineno)
        logger.log(
            level,
            "Unused kwargs {}. (called by {})".format(
                kwargs.keys(), called_by
            )
        )


def line_count(file: Path):
    """Efficient line counting for large file (for example large dataset file)

    Args:
        file (Path): _description_

    Returns:
        _type_: _description_
    """
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with file.open('r') as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


def batch_to_device(batch: MutableMapping[str, Any], device: torch.device) -> Dict[str, Any]:
    ret = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            v = v.to(device)
        ret[k] = v

    return ret
