"""
FlagGems Experimental Ops

This module provides experimental operators that are under development
and validation. These operators may not be fully optimized or tested
across all platforms.

Warning:
    Experimental operators are subject to change and may not have the
    same stability guarantees as stable operators in flag_gems.ops.

Usage:
    import flag_gems.experimental as fg_exp

    # Use operators directly
    from flag_gems.experimental.generated.pointwise import huber_loss
    result = huber_loss(input, target, reduction=1, delta=1.0)

    # Or access via module path
    result = fg_exp.generated.pointwise.huber_loss(input, target, reduction=1, delta=1.0)
"""

from __future__ import annotations

__version__ = "0.1.0"

# Import submodules
from . import generated
from . import custom
from . import metadata

__all__ = [
    "__version__",
    "generated",
    "custom",
    "metadata",
]
