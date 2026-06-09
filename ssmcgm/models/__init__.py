"""SSM-CGM forecasting model."""

__all__ = ["SSMCGM", "SSMCGMFast"]


def __getattr__(name):
    if name in __all__:
        from . import ssmcgm as _m
        return getattr(_m, name)
    raise AttributeError(name)
