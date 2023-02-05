from typing import List

from ..util.extention.rich import full_columns, no_total_columns
from . import ItrDataPipeline, LstDataPipeline, DataPipeline
from rich.progress import Progress


@DataPipeline.register()
class ItrToLst(LstDataPipeline):
    def __init__(
        self,
        datapipe: ItrDataPipeline,
        is_sized: bool,
        **kwargs
    ):
        super().__init__()
        self.data = []

        columns = full_columns() if is_sized else no_total_columns()
        total = len(datapipe) if is_sized else float('inf')
        with Progress(*columns) as pbar:
            tid = pbar.add_task(ItrToLst.__name__, total=total)
            for data in datapipe:
                self.data.append(data)
                pbar.advance(tid)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@DataPipeline.register()
class SequenceWrapper(LstDataPipeline):
    def __init__(self, datapipe: List, **kwargs):
        super().__init__()
        self.datapipe = datapipe

    def __getitem__(self, index):
        return self.datapipe[index]

    def __len__(self):
        return len(self.datapipe)
