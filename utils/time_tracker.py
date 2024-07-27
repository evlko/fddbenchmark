import time
from functools import wraps


def time_tracker(return_time=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(
                f"Function '{func.__name__}' took {elapsed_time:.4f} seconds to execute."
            )
            if return_time:
                return elapsed_time, result
            return result

        return wrapper

    return decorator
