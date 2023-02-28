"""
An inference helper for loading a checkpt and perform inference.
An AlchemyInferenceRunner will be created in a subprocess.

Communication protocol:
    - The parent process will send a List of data to start inference.
    - The child process will receive the data and perform inference.
    - The child returns results in order and terminates with a `None`.
    - The parent will send a `None` to the child to terminate the child process.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Tuple
from pathlib import Path

import torch
import multiprocessing as mp

from ..sym import sym_tbl
from ...plugins import AlchemyPlugin
from ...runner import get_dataloader
from ...model import AlchemyModel
from ...task import AlchemyTask
from ...pipeline.lst import SequenceWrapper
from ... import prepare_cfg


def _alchemy_nlp_task(
        conn,
        **kwargs
    ):
    with _AlchemyNLPRunner(**kwargs) as nlp:
        while True:
            data = conn.recv()
            if data is None:
                # main process sends None to terminate the child process
                break
            for result in nlp.pipe(data):
                conn.send(result)
            conn.send(None)     # send None to indicate the end of the data


class AlchemyNLP:
    def __init__(
            self,
            checkpt: Path,
            device_info: Dict[str, Any],
            bsz: int = 1,
            cfg: Optional[MutableMapping] = None,
            **kwargs
    ) -> None:
        conn, child_conn = mp.Pipe()
        self.conn = conn
        ctx = mp.get_context('spawn')

        self.p = ctx.Process(
            target=_alchemy_nlp_task,
            args=(child_conn, ),
            kwargs={
                "checkpt": checkpt,
                "device_info": device_info,
                "bsz": bsz,
                "cfg": cfg,
                **kwargs,
            }
        )
        self.p.start()

    def close(self):
        self.conn.send(None)
        self.p.join()

    def pipe(self, data: List) -> Iterator:
        if not isinstance(data, List):
            raise ValueError("The data must be a List")
        if len(data) == 0:
            return
        self.conn.send(data)
        while True:
            result = self.conn.recv()
            if result is None:
                break
            log, outputs = result
            if not isinstance(outputs, List):
                raise ValueError("The model output must be a List")
            for output in outputs:
                yield output


class _AlchemyNLPRunner:
    def __init__(
            self,
            checkpt: Path,
            device_info: Dict[str, Any],
            bsz: int = 1,
            cfg: Optional[MutableMapping] = None,
            **kwargs
    ) -> None:
        kwargs["no_file"] = True        # disable file logging

        # 1. load cfg
        if cfg is None:
            cfg = prepare_cfg(checkpt.parent.parent / "cfg.toml")       # Assume the checkpt is stored in "xxx/checpt/best"
            plugins = []
            for plugin in cfg["plugins"]:
                if plugin["type"] == "alchemy.plugins.BasicSetup":
                    # NOTE: plugins other than BasicSetup are not supported
                    plugins.append(plugin)
                    break
            cfg["plugins"] = plugins

            pipes = []
            for i, pipe in enumerate(cfg["task"]["datasets"]["dev"]["pipes"]):
                if i == 0:
                    # NOTE: We assume that the first pipeline loads data from files
                    # Here we replace it as a SequenceWrapper to load data dynamically
                    # If you violate this assumption, you need to pass `cfg` as an argument
                    pipe = {
                        "type": "alchemy.pipeline.lst.SequenceWrapper",
                        "datapipe": [],
                    }
                elif pipe["type"] == "alchemy.pipeline.lst.SequenceWrapper":
                    raise RuntimeError("\"alchemy.pipeline.lst.SequenceWrapper\" should only appear at the beginning of the pipeline.")
                if pipe["type"] == "alchemy.pipeline.lst.ItrToLst":
                    # Fully pipeline
                    continue
                if pipe["type"] == "alchemy.pipeline.itr.Batch":
                    pipe["batch_size"] = bsz
                pipes.append(pipe)
            cfg["task"]["datasets"]["inference"] = cfg["task"]["datasets"]["dev"]
            cfg["task"]["datasets"]["inference"]["pipes"] = pipes

            pipes = []
            for i, pipe in enumerate(cfg["task"]["outputpipes"]):
                if pipe["type"] == "alchemy.pipeline.output.Collect":
                    # NOTE: Output is directly returned, and there is no need to store them in sym_tbl
                    continue
                pipes.append(pipe)
            cfg["task"]["outputpipes"] = pipes

        sym_tbl().cfg = cfg
        sym_tbl().device_info = device_info
        sym_tbl().device = torch.device(device_info["device"])
        sym_tbl().ctime = datetime.now()

        for key, val in kwargs.items():
            sym_tbl().set_global(key, val)

        # plugins
        for p_cfg in sym_tbl().cfg.get("plugins", []):
            sym_tbl().plugins.append(AlchemyPlugin.from_registry(p_cfg["type"], **p_cfg))

        sym_tbl().task = AlchemyTask.from_registry(sym_tbl().cfg["task"]["type"])
        sym_tbl().model = AlchemyModel.from_registry(sym_tbl().cfg["model"]["type"])
        sym_tbl().model.to(sym_tbl().device)     # occupy GPU as soon as possible
        sym_tbl().model.eval()

        # There is no ItrToLst in pipelines
        sym_tbl().task.load_dataset(split="inference", **cfg["task"]["datasets"]["inference"])

    def __enter__(self) -> _AlchemyNLPRunner:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in sym_tbl().plugins:
            p.exit()

    def pipe(self, data: List) -> Iterator[Tuple[str, Any]]:
        """_summary_

        Args:
            data (List): _description_
            bsz (int): _description_

        Raises:
            NotImplementedError: _description_

        Yields:
            Iterator[str, Any]: _description_
        """
        dset, dset_kw = sym_tbl().task.dataset("inference")
        pipeline = dset
        while not isinstance(pipeline, SequenceWrapper):
            pipeline = pipeline.datapipe
        pipeline.datapipe = data

        itr = get_dataloader(
            dset,
            **dset_kw,
        )
        with torch.no_grad():
            for batch in itr:
                yield sym_tbl().task.eval_step(
                    batch,
                    needs_loss=False,
                )
