# Minimal init — only loggers and version imported at package level.
# TF1 and algo imports removed for NumPy 2.x / gymnasium compatibility.
try:
    from spinup.utils.logx import Logger, EpochLogger
    from spinup.version import __version__
except Exception:
    pass
