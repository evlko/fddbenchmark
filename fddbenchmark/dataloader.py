from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from fddbenchmark.config import DATA_FOLDER
from fddbenchmark.dataset import FDDDataset
from utils.time_tracker import time_tracker


class FDDDataloader(ABC):
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
        if dataset.df.index.names != ["run_id", "sample"]:
            raise ValueError("``dataframe`` must have multi-index ('run_id', 'sample')")

        if not np.all(dataset.df.index == dataset.train_mask.index) or not np.all(
            dataset.df.index == dataset.label.index
        ):
            raise ValueError("``dataframe`` and ``label`` must have the same indices.")

        if window_size <= 0 or step_size <= 0:
            raise ValueError(
                "``step_size`` and ``window_size`` must be positive numbers."
            )

        if step_size > window_size:
            raise ValueError("``step_size`` must be less or equal to ``window_size``.")

        if use_minibatches and batch_size is None:
            raise ValueError(
                "If you set ``use_minibatches=True``, "
                "you must set ``batch_size`` to a positive number."
            )

        self.dataset = dataset
        self.path = f"{DATA_FOLDER}/{self.dataset.name}/dataset.csv"
        self.window_size = window_size
        self.dilation = dilation
        self.step_size = step_size

        self.window_end_indices = self.get_windows(
            dataset=dataset,
            window_size=window_size,
            step_size=step_size,
            shuffle=shuffle,
            random_state=random_state,
            train=train,
        )

        n_samples = len(self.window_end_indices)
        batch_seq = list(range(0, n_samples, batch_size)) if use_minibatches else [0]
        batch_seq.append(n_samples)
        self.batch_seq = np.array(batch_seq)
        self.n_batches = len(batch_seq) - 1

    @staticmethod
    def get_mask(dataset: FDDDataset, train: bool) -> pd.Series:
        return dataset.train_mask if train else ~dataset.train_mask

    def get_windows(
        self,
        dataset: FDDDataset,
        window_size: int,
        step_size: int,
        shuffle: bool,
        random_state: int,
        train: bool,
    ):
        window_end_indices = []

        run_ids = (
            dataset.df[self.get_mask(dataset=dataset, train=train)]
            .index.get_level_values(0)
            .unique()
        )
        for run_id in tqdm(run_ids, desc="Creating sequence of samples"):
            indices = np.array(dataset.df.index.get_locs([run_id]))
            indices = indices[window_size - 1 :]
            indices = indices[::step_size]
            indices = indices[
                self.get_mask(dataset=dataset, train=train)
                .iloc[indices]
                .to_numpy(dtype=bool)
            ]
            window_end_indices.extend(indices)

        if random_state is not None:
            np.random.seed(random_state)

        window_end_indices = (
            np.random.permutation(window_end_indices)
            if shuffle
            else np.array(window_end_indices)
        )

        return window_end_indices

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        self.iter = 0
        return self

    def __next__(self):
        if self.iter < self.n_batches:
            ts_batch, index_batch, label_batch = self.__getitem__(self.iter)
            self.iter += 1
            return ts_batch, index_batch, label_batch
        else:
            raise StopIteration

    def __getitem__(self, idx):
        ends_indices = self.window_end_indices[
            self.batch_seq[idx] : self.batch_seq[idx + 1]
        ]
        windows_indices = (
            ends_indices[:, None] - np.arange(0, self.window_size, self.dilation)[::-1]
        )
        ts_batch = self.get_batch(windows_indices)
        label_batch = self.dataset.label.values[ends_indices]
        index_batch = self.dataset.label.index[ends_indices]

        return ts_batch, index_batch, label_batch

    @abstractmethod
    def get_batch(self, indices: np.ndarray) -> pd.DataFrame:
        pass
