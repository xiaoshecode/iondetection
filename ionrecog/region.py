"""Compatibility alias. Edit ionrecog.classical.region for implementation."""
import sys as _sys
from .classical import region as _implementation

_sys.modules[__name__] = _implementation
