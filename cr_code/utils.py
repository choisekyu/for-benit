import time


def check_time(func):
    def wrapper(*args, **kwargs):
        s = time.time()
        result = func(*args, **kwargs)
        print('\x1b[7m'+func.__name__, time.time() - s, '\x1b[0m')
        return result
    return wrapper
