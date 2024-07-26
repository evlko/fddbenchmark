import os
import zipfile
from typing import List

import pandas as pd
import requests
from tqdm.auto import tqdm

from fddbenchmark.config import AVAILABLE_DATASETS, DATA_FOLDER, SERVER_URL


class FDDDataset:
    def __init__(self, name: str):
        self.name = name
        available_datasets_str = ", ".join(AVAILABLE_DATASETS)
        if self.name not in AVAILABLE_DATASETS:
            raise Exception(
                f"{name} is an unknown dataset. Available datasets are: {available_datasets_str}"
            )

        ref_path = f"{DATA_FOLDER}/{self.name}/"
        if not os.path.exists(ref_path):
            os.makedirs(ref_path)
        url = f"{SERVER_URL}/{self.name}.zip"
        zfile_path = f"{DATA_FOLDER}/{self.name}.zip"
        if not os.path.exists(zfile_path):
            download_pgbar(url, zfile_path, fname=f"{self.name}.zip")

        extracting_files(zfile_path, ref_path)
        self.df = read_csv_pgbar(
            ref_path + "dataset.csv", index_col=["run_id", "sample"]
        )
        self.label = read_csv_pgbar(
            ref_path + "labels.csv", index_col=["run_id", "sample"]
        )["labels"]
        train_mask = read_csv_pgbar(
            ref_path + "train_mask.csv", index_col=["run_id", "sample"]
        )["train_mask"]
        test_mask = read_csv_pgbar(
            ref_path + "test_mask.csv", index_col=["run_id", "sample"]
        )["test_mask"]
        self.train_mask = train_mask.astype("boolean")
        self.test_mask = test_mask.astype("boolean")


def extracting_files(zfile_path: str, ref_path: str, bsize: int = 1024 * 10000) -> None:
    with zipfile.ZipFile(zfile_path, "r") as zfile:
        for entry_info in zfile.infolist():
            if os.path.exists(ref_path + entry_info.filename):
                continue
            input_file = zfile.open(entry_info.filename)
            target_file = open(ref_path + entry_info.filename, "wb")
            block = input_file.read(bsize)
            with tqdm(
                total=entry_info.file_size,
                desc=f"Extracting {entry_info.filename}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                while block:
                    target_file.write(block)
                    block = input_file.read(bsize)
                    pbar.update(bsize)
            input_file.close()
            target_file.close()


def download_pgbar(
    url: str, zfile_path: str, fname: str, chunk_size: int = 1024
) -> None:
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("Content-Length"))
    with open(zfile_path, "wb") as file:
        with tqdm(
            total=total,
            desc=f"Downloading {fname}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in resp.iter_content(chunk_size=chunk_size):
                file.write(data)
                pbar.update(len(data))


def read_csv_pgbar(
    csv_path: str, index_col: List[str], chunksize: int = 1024 * 100
) -> pd.DataFrame:
    rows = sum(1 for _ in open(csv_path, "r")) - 1
    chunk_list = []
    with tqdm(total=rows, desc=f"Reading {csv_path}") as pbar:
        for chunk in pd.read_csv(csv_path, index_col=index_col, chunksize=chunksize):
            chunk_list.append(chunk)
            pbar.update(len(chunk))
    df = pd.concat((f for f in chunk_list), axis=0)
    return df
