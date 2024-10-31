try:
    from line_profiler_pycharm import profile
except ImportError:
    def profile(func):
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner
