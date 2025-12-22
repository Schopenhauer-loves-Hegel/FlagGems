"""
empty_permuted - Experimental Implementation

This operator was automatically imported from generated code.

Metadata:
    op_id: 3d8c0114-80a3-4f04-bee3-225bd62d2608
    category: pointwise
    status: EXPERIMENTAL
    generator_tool: auto_codegen
    generation_date: 2025-12-22T11:54:47.424275

Warning:
    This is an experimental operator and may not be fully optimized or
    tested across all platforms. Use with caution.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def empty_permuted_kernel(
    out_ptr,  # Pointer to output tensor data
    n_elements,  # Number of elements in tensor
    BLOCK_SIZE: tl.constexpr,  # Block size for grid launch
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Intentionally do nothing to preserve "empty" (uninitialized) semantics.


def _compute_permuted_strides(size, permutation):
    # Compute strides such that the tensor is contiguous when iterating in `permutation` order.
    dim = len(size)
    assert len(permutation) == dim, "permutation must have the same number of dims as size"
    stride = [0] * dim
    curr = 1
    for idx in range(dim - 1, -1, -1):
        d = permutation[idx]
        s = int(size[d])
        if s == 0:
            s = 1
        stride[d] = curr
        curr *= s
    return tuple(stride)


def _parse_args_empty_permuted(args, kwargs):
    # Flexible parsing for factory-like signature: (size, permutation, *, dtype=None, device=None, layout=None, pin_memory=None)
    size = kwargs.get("size", None)
    permutation = kwargs.get("permutation", None)
    dtype = kwargs.get("dtype", None)
    device = kwargs.get("device", None)
    layout = kwargs.get("layout", None)
    pin_memory = kwargs.get("pin_memory", False)

    # Positional fallback
    if size is None and len(args) > 0:
        size = args[0]
    if permutation is None and len(args) > 1:
        permutation = args[1]
    if dtype is None and len(args) > 2 and isinstance(args[2], torch.dtype):
        dtype = args[2]
    if device is None and len(args) > 3 and isinstance(args[3], (torch.device, str, int)):
        device = args[3]
    if layout is None and len(args) > 4:
        layout = args[4]
    if len(args) > 5:
        pin_memory = args[5]

    if dtype is None:
        dtype = torch.float32

    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return size, permutation, dtype, device, layout, pin_memory


def empty_permuted(*args, **kwargs):
    size, permutation, dtype, device, layout, pin_memory = _parse_args_empty_permuted(args, kwargs)
    assert size is not None and permutation is not None, "size and permutation must be provided"

    strides = _compute_permuted_strides(list(size), list(permutation))
    out = torch.empty_strided(size=list(size), stride=strides, dtype=dtype, device=device)

    if out.is_cuda and out.numel() > 0:
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        empty_permuted_kernel[grid](out, n_elements, BLOCK_SIZE=1024)

    return out


def empty_permuted_out(*args, **kwargs):
    # Expected signature similar to: (out, size, permutation, *, dtype=None, device=None, layout=None, pin_memory=None)
    out = kwargs.get("out", None)
    positional = list(args)

    if out is None and len(positional) > 0 and isinstance(positional[0], torch.Tensor):
        out = positional.pop(0)

    size, permutation, dtype, device, layout, pin_memory = _parse_args_empty_permuted(positional, kwargs)
    assert size is not None and permutation is not None, "size and permutation must be provided"

    if out is None:
        # Fallback: create a new tensor if 'out' is not provided
        strides = _compute_permuted_strides(list(size), list(permutation))
        out = torch.empty_strided(size=list(size), stride=strides, dtype=dtype, device=device)
    else:
        # Resize the provided 'out' to requested size; cannot change strides in-place.
        if list(out.size()) != list(size):
            out.resize_(list(size))
        # Ensure dtype/device, move if needed
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype)
        if device is not None and out.device != torch.device(device):
            out = out.to(device)

    if out.is_cuda and out.numel() > 0:
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        empty_permuted_kernel[grid](out, n_elements, BLOCK_SIZE=1024)

    return out
