import sys
from typing import Optional

import numpy as np

from fddbenchmark import FDDDataloader, FDDDataset
from fddbenchmark.config import ICOLS
from fddbenchmark.dataset import read_csv_pgbar
from utils.time_tracker import time_tracker


class FDDPandasDataloader(FDDDataloader):
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
        self.df = read_csv_pgbar(
            csv_path=self.path,
            index_col=ICOLS,
        )

    @time_tracker(return_time=False)
    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        ts_batch = self.df.values[indices]  # (batch_size, window_size, ts_dim)
        return ts_batch
