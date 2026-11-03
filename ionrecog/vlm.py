"""Compatibility alias. Edit ionrecog.vision_language.pipeline for implementation."""
import sys as _sys
from .vision_language import pipeline as _implementation

_sys.modules[__name__] = _implementation
