from typing import Optional

from fddbenchmark import FDDDataloader, FDDDataset
import numpy as np
import pandas as pd
import polars as pl

from fddbenchmark.config import DATA_FOLDER


class PolarsDataloader(FDDDataloader):
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
        self.df = (pl.read_csv(path)).to_numpy()

    def get_batch(self,  indicies: np.ndarray) -> np.array:
        ts_batch = np.stack([self.df[idx] for idx in indicies], axis=0)
        return ts_batch
