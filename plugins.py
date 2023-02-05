# __future__.annotations will become the default in Python 3.11
from __future__ import annotations
from typing import List, Optional
import random
import datetime
import os
import shutil
from pathlib import Path

import numpy as np

from rich.logging import RichHandler
from rich.console import Console
from loguru import logger

import torch
from torch.utils.tensorboard import SummaryWriter

from .registry import Registrable
from .util.sym import sym_tbl


class AlchemyPlugin(Registrable):
    @classmethod
    def from_registry(cls, ty: str, **kwargs) -> AlchemyPlugin:
        plugin_cls = cls.resolve_registered_module(ty)
        plugin = plugin_cls(**kwargs)
        return plugin

    def __init__(self) -> None:
        super().__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


@AlchemyPlugin.register("BasicSetup")
class BasicSetup(AlchemyPlugin):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __enter__(self):
        # Some basic setup
        in_debug_mode = sym_tbl().try_get_global("debug", False)
        if in_debug_mode:
            from rich.traceback import install
            # install(show_locals=in_debug_mode)
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # config logger
        logger.remove()  # remove default stdout logger
        sym_tbl().console = Console()
        logger.add(
            RichHandler(markup=True, console=sym_tbl().console),
            level="DEBUG" if in_debug_mode else "INFO",
            # rich handler已经自带了时间、level和代码所在行
            format="[bold blue]" + sym_tbl().cfg["tag"] + "[/] - {message}",
        )


@AlchemyPlugin.register("FileLogger")
class FileLogger(AlchemyPlugin):
    def __init__(
        self,
        log_dir: str,
        subdirs: List[str] = [],
        create_readme: bool = True,
        backup_cfg: bool = True,
        **kwargs
    ) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)
        self.subdirs = subdirs
        self.create_readme = create_readme
        self.backup_cfg = backup_cfg

    def __enter__(self):
        import tomlkit

        no_file = sym_tbl().try_get_global("no_file", False)
        in_debug_mode = sym_tbl().try_get_global("debug", False)
        desc = sym_tbl().try_get_global("desc", "")

        if no_file:
            logger.info("Run without log and record due to no_file=True")
        else:
            # logging dir
            timestamp = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-')
            record_dir = self.log_dir / timestamp
            logger.info("Create record dir at \"{}\"", record_dir)
            record_dir.mkdir(parents=True, exist_ok=False)  # should not have same datetime.now()
            logger.add(
                str(record_dir / "run.log"),
                level="DEBUG" if in_debug_mode else "INFO",
            )
            checkpt_dir = record_dir / "checkpt"
            checkpt_dir.mkdir(parents=True)

            for subdir in self.subdirs:
                subdir: Path = record_dir / subdir
                subdir.mkdir(parents=True, exist_ok=True)

            if self.create_readme:
                # generate a doc file
                with (record_dir/ "README.md").open("w", encoding="utf8") as f:
                    f.write("# {}\n\n{}\n".format(sym_tbl().cfg["tag"], desc))

            if self.backup_cfg:
                backup_dir = record_dir / "backup"
                backup_dir.mkdir(parents=True, exist_ok=True)
                with (record_dir / "cfg.toml").open("w", encoding="utf8") as f:
                    tomlkit.dump(sym_tbl().cfg, f)  # backup config
            sym_tbl().set_global("record_dir", record_dir)
            sym_tbl().set_global("checkpt_dir", checkpt_dir)


@AlchemyPlugin.register("Backup")
class Backup(AlchemyPlugin):
    def __init__(self, paths: List[str], **kwargs) -> None:
        super().__init__()
        self.paths = [Path(p) for p in paths]

    def __enter__(self):
        if sym_tbl().contains_global("record_dir"):
            record_dir: Path = sym_tbl().get_global("record_dir")
            backup_dir = record_dir / "backup"
            backup_dir.mkdir(parents=True, exist_ok=True)
            for path in self.paths:
                if path.is_dir():
                    shutil.copytree(str(path), str(backup_dir / path.name))
                elif path.is_file():
                    shutil.copy(str(path), str(backup_dir / path.name))
                else:
                    raise NotImplementedError()


@AlchemyPlugin.register("TensorboardLogger")
class TensorboardLogger(AlchemyPlugin):

    def __init__(self, varname: str = "summary_writer", **kwargs) -> None:
        super().__init__()
        self.varname = varname
        self.summary_writer = None

    def __enter__(self):
        record_dir: Optional[Path] = sym_tbl().try_get("record_dir")
        if record_dir is not None:
            self.summary_writer = SummaryWriter(
                str(record_dir)
            )
            if not sym_tbl().try_set_global(self.varname, self.summary_writer):
                raise RuntimeError(f"\"{self.varname}\" already exists, there might be something wrong")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.summary_writer is not None:
            self.summary_writer.close()
            sym_tbl().pop_global(self.varname)


@AlchemyPlugin.register("Seeding")
class Seeding(AlchemyPlugin):
    def __init__(
        self,
        seed: int,
        use_deterministic_algorithms: bool = False,
        **kwargs
    ) -> None:
        super().__init__()
        self.seed = seed
        self.use_deterministic_algorithms = use_deterministic_algorithms

    def __enter__(self):
        sym_tbl().set_global("seed", self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if self.use_deterministic_algorithms:
            if torch.__version__ < "1.9":
                logger.warning(
                    "torch.use_deterministic_algorithms only support pytorch.__version__ >= 1.9. "
                    f"Your torch.__version__ = {torch.__version__}. "
                    "So use_deterministic_algorithms=true won't take effect."
                )
            else:
                torch.use_deterministic_algorithms(True)


@AlchemyPlugin.register("DisplayRunningInfo")
class DisplayRunningInfo(AlchemyPlugin):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def __enter__(self):
        logger.info("Alloc task \"{}\" on device {})".format(sym_tbl().cfg["tag"], sym_tbl().device_info))
        # 如果没有CUDA会有问题吗?
        logger.info(f"TorchVer={torch.__version__}(CUDA={torch.version.cuda},cuDNN={torch.backends.cudnn.version()})")
        logger.info(sym_tbl().cfg)
