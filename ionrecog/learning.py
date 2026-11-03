"""Compatibility alias. Edit ionrecog.self_supervised.learning for implementation."""
import sys as _sys
from .self_supervised import learning as _implementation

_sys.modules[__name__] = _implementation
