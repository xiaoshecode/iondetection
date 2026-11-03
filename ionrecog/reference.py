"""Compatibility alias. Edit ionrecog.classical.reference for implementation."""
import sys as _sys
from .classical import reference as _implementation

_sys.modules[__name__] = _implementation
