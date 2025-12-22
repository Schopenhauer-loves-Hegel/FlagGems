"""
log2_ - Experimental Implementation

This operator was automatically imported from generated code.

Metadata:
    op_id: 57c7ece9-8820-4a55-b0c1-35b1698da3e0
    category: pointwise
    status: EXPERIMENTAL
    generator_tool: auto_codegen
    generation_date: 2025-12-22T11:54:47.405384

Warning:
    This is an experimental operator and may not be fully optimized or
    tested across all platforms. Use with caution.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def log2_(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    val = tl.load(x_ptr + offsets, mask=mask)
    x = val.to(tl.float32)

    inv_ln2 = tl.full((), 1.4426950408889634, tl.float32)  # 1 / ln(2)
    y = tl.log(x) * inv_ln2

    out = y.to(val.dtype)
    tl.store(x_ptr + offsets, out, mask=mask)


# Keep a reference to the Triton kernel before redefining the name for the Python wrapper.
_log2__kernel = log2_


def log2_(*args, **kwargs):
    x = args[0] if len(args) > 0 else kwargs.get('input', None)
    if x is None:
        raise ValueError("log2_ expects a tensor as the first argument.")
    if not isinstance(x, torch.Tensor):
        raise TypeError("log2_ expects a torch.Tensor as input.")

    # Handle empty tensors directly
    if x.numel() == 0:
        return x

    # Fallback for non-CUDA tensors or unsupported dtypes
    if (not x.is_cuda) or (x.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        # Use PyTorch's implementation as a fallback
        x.log2_()
        return x

    # Work on a contiguous buffer; copy back if needed
    x_contig = x if x.is_contiguous() else x.contiguous()

    n_elements = x_contig.numel()
    if n_elements == 0:
        return x

    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _log2__kernel[grid](x_contig, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    if x_contig is not x:
        x.copy_(x_contig)

    return x
