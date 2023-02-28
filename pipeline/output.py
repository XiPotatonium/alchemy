import json
from pathlib import Path
from typing import Any, MutableMapping, Optional, Iterator, Union

from ..util.sym import sym_tbl
from ..util.json import NpJsonEncoder
from ..sched import AlchemyTrainScheduler
from . import OutputPipeline


@OutputPipeline.register()
class SaveAppend(OutputPipeline):
    def __init__(self, filename: str, step_tag: bool = True, **kwargs):
        super().__init__()
        self.filename = filename
        self.step_tag = step_tag

    def __call__(self, outputs: Union[Dict[str, Any], List], inputs: MutableMapping[str, Any]) -> Any:
        record_dir: Optional[Path] = sym_tbl().record_dir
        if record_dir is not None:
            filename = record_dir / self.filename
            sched: Optional[AlchemyTrainScheduler] = sym_tbl().train_sched
            if sched is not None and self.step_tag:
                # Path.with_stem not available under python 3.9
                filename = filename.with_name(
                    filename.stem + "_step{}".format(sched.cur_step) + filename.suffix
                )
            if not filename.parent.exists():
                filename.parent.mkdir(parents=True)
            with filename.open('a', encoding="utf8") as f:
                for output in outputs:
                    f.write(json.dumps(output, ensure_ascii=False, cls=NpJsonEncoder))
                    f.write('\n')
        return outputs


@OutputPipeline.register()
class Collect(OutputPipeline):

    def __init__(self, varname: str, **kwargs):
        super().__init__()
        self.varname = varname

    def __call__(self, outputs: Union[Dict[str, Any], List], inputs: MutableMapping[str, Any]) -> Any:
        sym_tbl().try_set_global(self.varname, [])
        sym_tbl().get_global(self.varname).extend(outputs)
        return outputs