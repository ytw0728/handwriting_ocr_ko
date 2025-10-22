import platform
import importlib
import sys

if platform.system() == "Darwin":
    _xp = importlib.import_module("numpy")
else:
    try:
        _xp = importlib.import_module("cupy")
    except ImportError:
        print("[warning] cupy not found; falling back to numpy")
        _xp = importlib.import_module("numpy")

globals().update(_xp.__dict__)
sys.modules[__name__] = _xp
