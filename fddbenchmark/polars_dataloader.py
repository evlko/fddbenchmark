import sys
from typing import Optional

from fddbenchmark import FDDDataloader, FDDDataset
import numpy as np
import pandas as pd
import polars as pl

from fddbenchmark.config import DATA_FOLDER
from utils.time_tracker import time_tracker


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
        self.df = pl.scan_csv(path)
        self.df = self.df.with_row_count("row_idx")
        print(sys.getsizeof(self.df))


    @time_tracker(return_time=False)
    def get_batch(self,  indicies: np.ndarray) -> np.array:
        all_indices = np.concatenate(indicies)
        filtered_df = self.df.filter(pl.col("row_idx").is_in(all_indices))

        collected_df = filtered_df.collect()

        df_np = collected_df.to_numpy()

        results = [df_np[idx] for idx in indicies]
        return results

