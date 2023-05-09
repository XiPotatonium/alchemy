import itertools
import random
from typing import Iterator, List, Union, Dict, Any

from pipeline import DataPipeline
from . import ItrDataPipeline, DataPipeline
from torch.utils.data import get_worker_info


@DataPipeline.register()
class Batch(ItrDataPipeline):
    def __init__(
        self,
        datapipe: Union[List, Dict[str, Any], DataPipeline],
        batch_size: int = 1,
        drop_last: bool = False,
        **kwargs,
    ):
        super().__init__(datapipe)
        self.require_single_source()
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for sample in iter(self.datapipe):
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            if not self.drop_last:
                yield batch


@DataPipeline.register()
class WithLength(ItrDataPipeline):
    def __init__(
        self,
        datapipe: Union[List, Dict[str, Any], DataPipeline],
        length: int,
        **kwargs,
    ):
        super().__init__(datapipe)
        self.require_single_source()
        self.length = length

    def __iter__(self) -> Iterator:
        return iter(self.datapipe)

    def __len__(self) -> int:
        return self.length


@DataPipeline.register()
class Shuffle(ItrDataPipeline):
    def __init__(
        self,
        datapipe: Union[List, Dict[str, Any], DataPipeline],
        buffer_size: int = 1000,
        **kwargs,
    ):
        super().__init__(datapipe)
        self.require_single_source()
        self.buffer_size = buffer_size
        self._shuffle_enabled = True

    def set_shuffle_settings(self, shuffle=True):
        self._shuffle_enabled = shuffle

    @staticmethod
    def buffer_replace(buffer, x):
        idx = random.randint(0, len(buffer) - 1)
        val = buffer[idx]
        buffer[idx] = x
        return val

    def __iter__(self) -> Iterator:
        if not self._shuffle_enabled:
            for x in self.datapipe:
                yield x
        else:
            buffer: List = []
            for x in self.datapipe:
                if len(buffer) == self.buffer_size:
                    yield Shuffle.buffer_replace(buffer, x)
                else:
                    buffer.append(x)
            random.shuffle(buffer)
            while buffer:
                yield buffer.pop()


@DataPipeline.register()
class SplitByWorker(ItrDataPipeline):
    def __init__(self, datapipe: Union[List, Dict[str, Any], DataPipeline], **kwargs):
        super().__init__(datapipe)
        self.require_single_source()

    def __iter__(self) -> Iterator:
        worker_info = get_worker_info()
        if worker_info is not None:
            return itertools.islice(
                self.datapipe, worker_info.id, None, worker_info.num_workers
            )
        else:
            return iter(self.datapipe)


@DataPipeline.register()
class Concat(ItrDataPipeline):
    def __init__(
        self, datapipe: Union[List, Dict[str, Any], DataPipeline], **kwargs
    ) -> None:
        super().__init__(datapipe)
        if isinstance(self.datapipe, List):
            pass
        elif isinstance(self.datapipe, Dict):
            self.datapipe = list(self.datapipe.values())
        elif isinstance(self.datapipe, DataPipeline):
            self.datapipe = [self.datapipe]
        else:
            raise ValueError("Invalid datapipe type {}".format(type(self.datapipe)))

    def __iter__(self) -> Iterator:
        return itertools.chain(self.datapipe)

    def __len__(self):
        return sum(len(pipe) for pipe in self.datapipe)


@DataPipeline.register()
class Sample(ItrDataPipeline):
    def __init__(
        self,
        datapipe: Union[List, Dict[str, Any], DataPipeline],
        ratios: Union[List, Dict[str, Any]],
        epoch_items: int,
        **kwargs,
    ) -> None:
        super().__init__(datapipe)
        self.epoch_items = epoch_items

        if isinstance(ratios, List):
            if not isinstance(datapipe, List):
                raise ValueError(
                    "Mismatch source data type {} and ratios type {}".format(
                        type(datapipe), type(ratios)
                    )
                )
            if len(ratios) != len(datapipe):
                raise ValueError(
                    "Ratio length {} mismatch, expect {}".format(
                        len(ratios), len(datapipe)
                    )
                )
            self.sample_keys = list(range(len(ratios)))
            self.datapipe = [itertools.cycle(pipe) for pipe in self.datapipe]
        elif isinstance(ratios, Dict):
            if not isinstance(datapipe, Dict):
                raise ValueError(
                    "Mismatch source data type {} and ratios type {}".format(
                        type(datapipe), type(ratios)
                    )
                )
            if len(ratios) != len(datapipe) or not all(
                k in ratios for k in datapipe.keys()
            ):
                raise ValueError(
                    "Ratio keys {} mismatch, expect {}".format(
                        ratios.keys(), datapipe.keys()
                    )
                )
            self.sample_keys = list(ratios.keys())
            self.ratios = list(ratios.values())
            self.datapipe = {k: itertools.cycle(v) for k, v in self.datapipe.items()}
        else:
            raise ValueError(
                "Invalid ratio type {}, expect list or dict".format(type(ratios))
            )

    def __iter__(self) -> Iterator:
        for _ in range(self.epoch_item):
            sampled_key = random.choices(self.sample_keys, weights=self.ratios, k=1)[0]
            yield next(self.datapipe[sampled_key])

    def __len__(self):
        return self.epoch_items
