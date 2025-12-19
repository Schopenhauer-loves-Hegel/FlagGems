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

    # Enable experimental operators globally
    fg_exp.enable()

    # Now torch will use experimental operators
    result = torch.ops.aten.huber_loss(input, target, reduction=1, delta=1.0)

    # Or use directly without enable()
    from flag_gems.experimental.generated.pointwise import huber_loss
    result = huber_loss(input, target, reduction=1, delta=1.0)

    # Disable when done
    fg_exp.disable()
"""

from __future__ import annotations

from typing import List, Optional
import torch

__version__ = "0.1.0"

# Global state
_experimental_lib: Optional[torch.library.Library] = None
_experimental_registrar = None

# Import submodules
from . import generated
from . import custom
from . import metadata


def enable(
    groups: Optional[List[str]] = None,
    lib: Optional[torch.library.Library] = None,
) -> None:
    """
    Enable experimental operators by registering them to torch.

    This function registers experimental operators to PyTorch's operator
    dispatch system, allowing them to be called via torch.ops.aten.*.

    Args:
        groups: List of operator groups to enable. Options:
                - None: Enable all operators (default)
                - ['generated']: Only generated operators
                - ['custom']: Only custom operators
                - ['generated', 'custom']: Both
        lib: torch.library.Library instance. If None, creates a new one.

    Example:
        >>> import flag_gems.experimental as fg_exp
        >>> fg_exp.enable()
        >>> # Now can use: torch.ops.aten.huber_loss(...)

    Note:
        Experimental operators will override stable operators if both exist.
        Call disable() to unregister experimental operators.
    """
    global _experimental_lib, _experimental_registrar

    # Import Register from main flag_gems
    from flag_gems.runtime.register import Register

    # Create or use provided Library
    if lib is None:
        _experimental_lib = torch.library.Library("aten", "IMPL")
        lib = _experimental_lib
    else:
        _experimental_lib = lib

    # Build operator registration list
    op_list = _build_op_registration_list(groups)

    if not op_list:
        print("⚠️  No experimental operators to register")
        return

    # Register using the same mechanism as main branch
    _experimental_registrar = Register(
        op_list,
        user_unused_ops_list=[],
        cpp_patched_ops_list=[],
        lib=lib,
    )

    print(f"✅ Enabled {len(op_list)} experimental operators")


def _build_op_registration_list(groups: Optional[List[str]]) -> tuple:
    """
    Build operator registration list dynamically from metadata.

    Args:
        groups: Operator groups to include

    Returns:
        Tuple of (op_name, op_impl) or (op_name, op_impl, condition) tuples
    """
    from pathlib import Path
    from .metadata import MetadataManager, OpCategory, OpStatus

    # Get metadata file path
    exp_root = Path(__file__).parent
    metadata_file = exp_root / "generated" / "_metadata.json"

    if not metadata_file.exists():
        print(f"⚠️  Metadata file not found: {metadata_file}")
        return tuple()

    # Load metadata
    metadata_mgr = MetadataManager(metadata_file)

    # Determine which categories to include based on groups
    categories = []
    if groups is None:
        # All groups
        categories = [OpCategory.POINTWISE, OpCategory.REDUCTION, OpCategory.BLAS, OpCategory.CUSTOM]
    else:
        if 'generated' in groups:
            categories.extend([OpCategory.POINTWISE, OpCategory.REDUCTION, OpCategory.BLAS])
        if 'custom' in groups:
            categories.append(OpCategory.CUSTOM)

    # Query operators
    op_list = []
    for category in categories:
        ops_metadata = metadata_mgr.query_ops({"category": category.value})

        for op_meta in ops_metadata:
            # Only include operators that are at least EXPERIMENTAL status
            if op_meta.status == OpStatus.EXPERIMENTAL or \
               op_meta.status == OpStatus.VALIDATED or \
               op_meta.status == OpStatus.GRADUATION_CANDIDATE:

                # Try to load the operator
                op_impl = _load_operator(op_meta)

                if op_impl is not None:
                    op_list.append((op_meta.op_name, op_impl))
                    print(f"  📦 Loaded: {op_meta.op_name} (category: {op_meta.category.value})")

    return tuple(op_list)


def _load_operator(op_metadata):
    """
    Dynamically load operator implementation.

    Args:
        op_metadata: OpMetadata object

    Returns:
        Operator implementation function or None if failed
    """
    try:
        from .metadata import OpCategory

        category = op_metadata.category.value
        op_name = op_metadata.op_name

        # Construct import path
        if op_metadata.category in [
            OpCategory.POINTWISE,
            OpCategory.REDUCTION,
            OpCategory.BLAS,
        ]:
            module_path = f"flag_gems.experimental.generated.{category}.{op_name}"
        else:
            module_path = f"flag_gems.experimental.custom.{op_name}"

        # Dynamic import
        module = __import__(module_path, fromlist=[op_name])
        op_impl = getattr(module, op_name)

        return op_impl

    except Exception as e:
        print(f"⚠️  Failed to load {op_metadata.op_name}: {e}")
        return None


def disable() -> None:
    """
    Disable experimental operators.

    Unregisters experimental operators from torch. After calling this,
    torch will use the default implementations.

    Example:
        >>> import flag_gems.experimental as fg_exp
        >>> fg_exp.enable()
        >>> # ... use experimental ops ...
        >>> fg_exp.disable()
    """
    global _experimental_lib, _experimental_registrar

    if _experimental_lib and hasattr(_experimental_lib, "_destroy"):
        try:
            if torch.__version__ >= "2.5":
                _experimental_lib._destroy()
        except Exception as e:
            print(f"⚠️  Error destroying library: {e}")

    _experimental_lib = None
    _experimental_registrar = None
    print("✅ Experimental operators disabled")


def list_enabled_ops() -> List[str]:
    """
    List currently enabled experimental operators.

    Returns:
        List of operator names that are currently registered

    Example:
        >>> import flag_gems.experimental as fg_exp
        >>> fg_exp.enable()
        >>> ops = fg_exp.list_enabled_ops()
        >>> print(ops)
        ['huber_loss']
    """
    if _experimental_registrar and hasattr(_experimental_registrar, "get_all_ops"):
        return _experimental_registrar.get_all_ops()
    return []


def is_enabled() -> bool:
    """
    Check if experimental operators are currently enabled.

    Returns:
        True if experimental operators are enabled, False otherwise
    """
    return _experimental_registrar is not None


__all__ = [
    "__version__",
    "enable",
    "disable",
    "list_enabled_ops",
    "is_enabled",
    "generated",
    "custom",
    "metadata",
]
