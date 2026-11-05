__all__ = ["state_detection",
           "io",
           "switchfunction",
           "skewloss",
           "model_skewencoder",
           "gen_plumed",
           "plumedkits"]

# Submodules and their names are imported on first access (PEP 562), not here.
# skewencoder.contactwall runs inside PLUMED's embedded Python (PyCV); importing torch there makes
# PYTORCH_MODEL's autograd call abort with "The autograd engine was called while holding the GIL".
import importlib

_SUBMODULES = [*__all__, "contactwall", "elements"]


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    for module_name in __all__:
        module = importlib.import_module(f".{module_name}", __name__)
        if name in getattr(module, "__all__", ()):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
