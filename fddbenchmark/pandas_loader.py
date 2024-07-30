from typing import Any, Optional

import pandas as pd
import numpy as np
from numpy import ndarray, dtype

from fddbenchmark import FDDDataloader, FDDDataset
from fddbenchmark.config import DATA_FOLDER

import polars as pl


class PandasLoader(FDDDataloader):
    def get_batch(self, indicies: np.ndarray) -> np.array:
        path = f"{DATA_FOLDER}/{self.dataset.name}/dataset.csv"
        df = pd.read_csv(path)
        flat_indices = indicies.flatten()
        data = df.iloc[flat_indices].to_numpy()
        res = data.reshape(indicies.shape[0], indicies.shape[1], data.shape[1])
        return res
