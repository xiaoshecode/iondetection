"""Compatibility alias. Edit ionrecog.classical.detection for implementation."""
import sys as _sys
from .classical import detection as _implementation

_sys.modules[__name__] = _implementation
