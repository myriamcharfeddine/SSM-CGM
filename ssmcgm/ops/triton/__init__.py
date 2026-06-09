"""Custom Triton kernels for the MES-Mamba2 scan."""

__all__ = ["mes_selective_scan_triton"]


def __getattr__(name):
    if name == "mes_selective_scan_triton":
        from .mes_mamba2_scan import mes_selective_scan_triton
        return mes_selective_scan_triton
    raise AttributeError(name)
