__version__ = "1.0.0"
__all__ = ["RRTANet", "create_rrta", "warp", "registered_upsample"]


def __getattr__(name):
    if name in __all__:
        from . import rrta

        return getattr(rrta, name)
    raise AttributeError(name)
