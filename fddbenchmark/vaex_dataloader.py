import vaex
import numpy as np
import sys
from typing import Optional

from fddbenchmark import FDDDataloader, FDDDataset
from utils.time_tracker import time_tracker


class FDDDVaexDataloader(FDDDataloader):
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
        self.df = vaex.open(self.path)
        self.df["row_idx"] = vaex.vrange(0, len(self.df))
        print(sys.getsizeof(self.df))

    @time_tracker(return_time=False)
    def get_batch(self, indices: np.ndarray) -> np.ndarray:
        all_indices = np.concatenate(indices)
        unique_indices = np.unique(all_indices)
        mask = self.df.row_idx.isin(unique_indices)
        filtered_df = self.df[mask]
        df_pd = filtered_df.to_pandas_df()
        df_pd = df_pd.iloc[:, 2:-1]
        df_np = df_pd.to_numpy()
        idx_map = {idx: i for i, idx in enumerate(unique_indices)}
        results = np.array([df_np[idx_map[idx]] for idx in all_indices])
        results = results.reshape(indices.shape + (-1,))
        return results
