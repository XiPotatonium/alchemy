from typing import List, Union, Dict, Any

from ..util.extention.rich import full_columns, no_total_columns
from . import ItrDataPipeline, LstDataPipeline, DataPipeline
from rich.progress import Progress


@DataPipeline.register()
class ItrToLst(LstDataPipeline):
    def __init__(
        self,
        datapipe: Union[List, Dict[str, Any], DataPipeline],
        is_sized: bool,
        **kwargs
    ):
        super().__init__(datapipe)
        self.require_single_source()

        self.datapipe = []
        columns = full_columns() if is_sized else no_total_columns()
        total = len(datapipe) if is_sized else float('inf')
        with Progress(*columns) as pbar:
            tid = pbar.add_task(ItrToLst.__name__, total=total)
            for data in datapipe:
                self.datapipe.append(data)
                pbar.advance(tid)

    def __getitem__(self, index):
        return self.datapipe[index]

    def __len__(self):
        return len(self.datapipe)


@DataPipeline.register()
class SequenceWrapper(LstDataPipeline):
    def __init__(self, datapipe: Union[List, Dict[str, Any], DataPipeline], **kwargs):
        super().__init__(datapipe)
        if not isinstance(self.datapipe, List):
            raise ValueError("Expect list data source but found {}".format(type(self.datapipe)))

    def __getitem__(self, index):
        return self.datapipe[index]

    def __len__(self):
        return len(self.datapipe)
