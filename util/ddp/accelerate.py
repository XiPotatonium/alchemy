import itertools
import os
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Sized, Union
from loguru import logger
from rich.logging import RichHandler
from rich.progress import Progress
from rich.console import Console
from torch import Tensor, nn
from torch.utils.data import IterableDataset
from alchemy.plugins import AlchemyPlugin
from alchemy.runner import evaluate, get_dataloader, AlchemyTrainer
from alchemy.model import BackwardHandler
from alchemy.pipeline import (
    DataPipeline, EvalPipeline, ItrDataPipeline, SchedPipeline, EndStepPipeline
)
from alchemy import sym_tbl, AlchemyTrainScheduler, AlchemyTask, AlchemyModel, AlchemyOptimizer, AlchemyRunner
from alchemy.util import filter_optional_cfg
from alchemy.util.extention.rich import full_columns, no_total_columns
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.optimization import AdamW as _HFAdamW
from transformers import PreTrainedModel


_VARNAME = "accelerator"


@AlchemyRunner.register()
class Trainer(AlchemyTrainer):
    def __init__(self, cfg: MutableMapping[str, Any], device_info: Dict[str, Any], **kwargs):
        accelerator = Accelerator(kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
        if not accelerator.is_main_process:
            # only do recording in main process
            kwargs["no_file"] = True
        sym_tbl().set_global(_VARNAME, accelerator)
        super().__init__(cfg, device_info, **kwargs)
        sym_tbl().device_info = accelerator.device

    def run(self):
        sym_tbl().task = AlchemyTask.from_registry(sym_tbl().cfg["task"]["type"])
        sym_tbl().model = AlchemyModel.from_registry(sym_tbl().cfg["model"]["type"])
        # sym_tbl().model.to(sym_tbl().device)     # occupy GPU as soon as possible

        # TODO: 一个优化，只有需要评估的main_process才需要加载dev和test数据集
        for split, dataset_cfg in sym_tbl().cfg["task"]["datasets"].items():
            sym_tbl().task.load_dataset(split=split, **dataset_cfg)

        sym_tbl().optim = AlchemyOptimizer.from_registry(sym_tbl().cfg["optim"]["type"])
        sym_tbl().train_sched = AlchemyTrainScheduler.from_registry(sym_tbl().cfg["sched"]["type"])

        # 将Model，optimizer，sched转换为accelerate需要的形式
        accelerator: Accelerator = sym_tbl().get_global(_VARNAME)
        model, optimizer, lr_sched = accelerator.prepare(
            sym_tbl().model.model,
            sym_tbl().optim.optimizer,
            sym_tbl().train_sched.lr_sched,
        )
        sym_tbl().model.model = model
        sym_tbl().optim.optimizer = optimizer
        sym_tbl().train_sched.lr_sched = lr_sched
        # TODO: 把这行放到数据集加载前面，但是似乎需要保证model to device要在accelerate.prepare的后面
        sym_tbl().model.to(sym_tbl().device)     # occupy GPU as soon as possible

        dset, dset_kw = sym_tbl().task.dataset('train')
        itr = get_dataloader(dset, **dset_kw)
        total = len(itr) if not isinstance(dset, IterableDataset) else float('inf')
        columns = full_columns() if not isinstance(dset, IterableDataset) else no_total_columns()

        for epoch in sym_tbl().train_sched:
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


@AlchemyPlugin.register()
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

        logger.remove()  # remove default logger
        accelerator: Accelerator = sym_tbl().get_global(_VARNAME)
        if accelerator.is_local_main_process or in_debug_mode:
            # config logger
            sym_tbl().console = Console()
            logger.add(
                RichHandler(markup=True, console=sym_tbl().console),
                level="DEBUG" if in_debug_mode else "INFO",
                # rich handler已经自带了时间、level和代码所在行
                format="[bold blue]" + sym_tbl().cfg["tag"] + "[/] - {message}",
            )


@DataPipeline.register()
class Sharding(ItrDataPipeline):
    def __init__(self, datapipe: ItrDataPipeline, **kwargs):
        super().__init__()
        self.datapipe = datapipe
        # NOTE: runner should not be a field of DataPipeline
        accelerator: Accelerator = sym_tbl().get_global(_VARNAME)
        self.node_count = accelerator.num_processes
        self.node_id = accelerator.process_index

    def __iter__(self) -> Iterator:
        if self.node_count != 1:
            return itertools.islice(self.datapipe, self.node_id, None, self.node_count)
        return iter(self.datapipe)


@AlchemyOptimizer.register()
class HFAdamW(AlchemyOptimizer):
    def __init__(self):
        super(HFAdamW, self).__init__()
        self.reset()

    def reset(self, **kwargs):
        # 如果kwargs里面没有，那么从self.cfg里面找，如果还是没有，则为默认值
        cfg = dict(self.cfg)
        cfg.update(kwargs)

        self.max_grad_norm = cfg["max_grad_norm"]

        self._optimizer = _HFAdamW(
            sym_tbl().model.optim_params(**cfg),
            **filter_optional_cfg(
                cfg=cfg,
                optional_keys={"lr", "weight_decay", "betas", "eps", "correct_bias"},
            ),
            no_deprecation_warning=True,
        )

    def step(self):
        accelerator: Accelerator = sym_tbl().get_global(_VARNAME)
        accelerator.clip_grad_norm_(self.params, self.max_grad_norm)
        super().step()


@BackwardHandler.register("Backward")
class BackwardHandle(BackwardHandler):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def backward(
        self,
        losses: Union[List[Tensor], Tensor],
        weights: Optional[List[Tensor]] = None,
        requires_grad: bool = True,
        names: Optional[List[str]] = None,
    ) -> float:
        losses = [losses] if isinstance(losses, Tensor) else losses
        if weights is None:
            loss = sum(losses)
        else:
            loss = sum(l * w for l, w in zip(losses, weights))

        if requires_grad:
            accelerator: Accelerator = sym_tbl().get_global(_VARNAME)
            sym_tbl().optim.zero_grad()
            accelerator.backward(loss)
            sym_tbl().optim.step()
            sym_tbl().train_sched.step_lr()

        return loss.item()


@SchedPipeline.register("EvalStepESPipeline")
class EvalStepESPipeline(EndStepPipeline):
    def __init__(self, period: int, split: str = "dev", needs_loss: bool = True, **kwargs):
        super().__init__()
        self.period = period
        self.split = split
        self.needs_loss = needs_loss

    def __call__(self, outputs: MutableMapping[str, Any], **kwargs) -> Dict[str, Any]:
        accelerator: Accelerator = sym_tbl().get_global(_VARNAME)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return kwargs
        step = sym_tbl().train_sched.cur_step
        if (step + 1) % self.period == 0:
            logger.info(
                "EvalStep at epoch {} step {}".format(
                    sym_tbl().train_sched.cur_epoch,
                    sym_tbl().train_sched.cur_step
                )
            )
            evaluate(self.split, needs_loss=self.needs_loss)

        return kwargs


class SaveModelMixin:
    def _prepare_to_save(self, model: Union[PreTrainedModel, nn.Module]):
        accelerator: Accelerator = sym_tbl().get_global(_VARNAME)
        return accelerator.unwrap_model(model)
