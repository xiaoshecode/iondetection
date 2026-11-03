"""Compatibility alias. Edit ionrecog.self_supervised.models for implementation."""
import sys as _sys
from .self_supervised import models as _implementation

_sys.modules[__name__] = _implementation
