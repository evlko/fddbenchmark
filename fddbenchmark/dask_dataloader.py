from typing import Optional

import dask.array as da
import dask.dataframe as dd
import numpy as np

from fddbenchmark.dataloader import FDDDataloader
from fddbenchmark.dataset import FDDDataset
from utils.time_tracker import time_tracker


class FDDDaskDataloader(FDDDataloader):
    def __init__(
        self,
        dataset: FDDDataset,
        train: bool,
        window_size: int,
        dilation: int = 1,
        step_size: int = 1,
        use_minibatches: bool = False,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset=dataset,
            train=train,
            window_size=window_size,
            dilation=dilation,
            step_size=step_size,
            use_minibatches=use_minibatches,
            batch_size=batch_size,
            shuffle=shuffle,
            random_state=random_state,
        )
        path = f"data/{self.dataset.name}/dataset.csv"
        ddf = dd.read_csv(path)
        ddf = ddf.drop(ddf.columns[:2].tolist(), axis=1)
        self.dask_array = ddf.to_dask_array(lengths=True)

    @time_tracker(return_time=False)
    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        print(indices)
        ts_batch = da.stack([self.dask_array[idx] for idx in indices], axis=0).compute()
        return ts_batch
