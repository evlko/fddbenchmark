from fddbenchmark import FDDDataset


def test_small_tep():
    dataset = FDDDataset(name="small_tep")
    assert dataset.df.shape == (153300, 0)
    assert dataset.label.shape == (153300,)
