import numpy as np

from fddbenchmark import FDDDataset
from fddbenchmark.dataloader import FDDDataloaderPandas


def test_small_tep():
    dataset = FDDDataset(name="small_tep")
    loader = FDDDataloaderPandas(
        dataset=dataset,
        train=True,
        window_size=100,
        step_size=1,
        use_minibatches=True,
        batch_size=1024,
    )
    assert len(loader) == 42
    for ts, time_index, label in loader:
        break
    assert ts.shape == (1024, 100, 0)
    assert len(time_index) == 1024
    assert label.shape == (1024,)
    # assert np.all(
    #    ts[0, 0, :5] == [2.5038e-01, 3.6740e03, 4.5290e03, 9.2320e00, 2.6889e01]
    # )
