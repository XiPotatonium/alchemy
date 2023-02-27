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
import datetime
from typing import Any, Dict, Iterator, List, MutableMapping, Optional
from pathlib import Path

import torch
import multiprocessing as mp
from multiprocessing.connection import PipeConnection

from plugins import AlchemyPlugin
from ..sym import sym_tbl
from ...runner import get_dataloader
from ...model import AlchemyModel
from ...task import AlchemyTask
from ... import prepare_cfg


def _alchemy_nlp_task(
        conn: PipeConnection,
        **kwargs
    ):
    with _AlchemyNLPRunner(**kwargs) as nlp:
        while True:
            data = conn.recv()
            if data is None:
                break
            for result in nlp.pipe(data):
                conn.send(result)


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

    def __enter__(self) -> _AlchemyNLPRunner:
        self.p.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.send(None)
        self.p.join()

    def pipe(self, data: List) -> Iterator:
        self.conn.send(data)
        while True:
            result = self.conn.recv()
            if result is None:
                break
            yield result


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
            cfg = prepare_cfg(checkpt.parent / "cfg.toml")
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

        # There is no ItrToLst in pipelines
        sym_tbl().task.load_dataset(split="inference")

    def __enter__(self) -> _AlchemyNLPRunner:
        for p in sym_tbl().plugins:
            p.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in sym_tbl().plugins:
            p.__exit__(exc_type, exc_val, exc_tb)

    def pipe(self, data: List) -> Iterator[str, Any]:
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
        while pipeline.type != "alchemy.pipeline.lst.SequenceWrapper":
            pipeline = pipeline.datapipe
        pipeline.datapipe = data

        itr = get_dataloader(
            dset,
            **dset_kw,
        )
        for batch in itr:
            yield sym_tbl().task.eval_step(
                batch,
                needs_loss=False,
            )
