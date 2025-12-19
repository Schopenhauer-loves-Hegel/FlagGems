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

    # Use generated experimental operators
    result = fg_exp.generated.some_op(x)

    # Use custom experimental operators
    result = fg_exp.custom.some_custom_op(x)
"""

from __future__ import annotations

__version__ = "0.1.0"

# Submodules will be imported as they are implemented
# from . import generated
# from . import custom
# from . import graduation

__all__ = [
    "__version__",
]
