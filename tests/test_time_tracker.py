import time

from utils.time_tracker import time_tracker


@time_tracker(return_time=True)
def example_function(seconds: int) -> None:
    return time.sleep(seconds)


def test_time_tracker():
    time, _ = example_function(2)
    assert round(time) == 2
