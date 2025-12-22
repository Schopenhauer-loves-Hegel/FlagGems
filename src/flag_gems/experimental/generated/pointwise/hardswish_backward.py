"""
hardswish_backward - Experimental Implementation

This operator was automatically imported from generated code.

Metadata:
    op_id: fda954d2-9725-49cf-91f5-a7a6405983c3
    category: pointwise
    status: EXPERIMENTAL
    generator_tool: auto_codegen
    generation_date: 2025-12-22T11:54:47.175427

Warning:
    This is an experimental operator and may not be fully optimized or
    tested across all platforms. Use with caution.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def hardswish_bwd_kernel(grad_out_ptr, x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    go = tl.load(grad_out_ptr + offsets, mask=mask)
    x = tl.load(x_ptr + offsets, mask=mask)

    mid = x / 3.0 + 0.5
    deriv = tl.where(x < -3.0, 0.0, tl.where(x <= 3.0, mid, 1.0))
    gout = go * deriv

    tl.store(out_ptr + offsets, gout, mask=mask)


def _check_inputs(grad_output: torch.Tensor, self: torch.Tensor, out: torch.Tensor = None):
    if grad_output.device.type != 'cuda' or self.device.type != 'cuda' or (out is not None and out.device.type != 'cuda'):
        raise RuntimeError("All tensors must be on CUDA device for Triton kernel execution.")
    if grad_output.numel() != self.numel():
        raise RuntimeError("grad_output and self must have the same number of elements.")
    if out is not None and out.numel() != self.numel():
        raise RuntimeError("out must have the same number of elements as inputs.")
    if grad_output.shape != self.shape:
        raise RuntimeError("grad_output and self must have the same shape.")
    if out is not None and out.shape != self.shape:
        raise RuntimeError("out must have the same shape as inputs.")
    allowed_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if grad_output.dtype not in allowed_dtypes or self.dtype not in allowed_dtypes:
        raise RuntimeError("Supported dtypes are float16, bfloat16, and float32.")


def _launch_hardswish_backward(grad_output: torch.Tensor, self: torch.Tensor, out: torch.Tensor):
    n_elements = out.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    hardswish_bwd_kernel[grid](
        grad_output, self, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )


def hardswish_backward(grad_output: torch.Tensor, self: torch.Tensor):
    _check_inputs(grad_output, self)
    go = grad_output.contiguous()
    x = self.contiguous()
    out = torch.empty_like(x)
    _launch_hardswish_backward(go, x, out)
    return out


def hardswish_backward_out(grad_output: torch.Tensor, self: torch.Tensor, out: torch.Tensor):
    _check_inputs(grad_output, self, out)
    go = grad_output.contiguous()
    x = self.contiguous()
    use_out = out if out.is_contiguous() else torch.empty_like(out)
    _launch_hardswish_backward(go, x, use_out)
    if use_out.data_ptr() != out.data_ptr():
        out.copy_(use_out)
    return out
