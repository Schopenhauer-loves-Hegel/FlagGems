import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _mish_fwd_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    sp = tl.log(1.0 + tl.exp(x))
    out = x * tl.math.tanh(sp)
    tl.store(out_ptr + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


def mish(self):
    logger.debug("GEMS_KUNLUNXIN MISH_FORWARD")
    output = torch.empty_like(self)
    n = self.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _mish_fwd_kernel[grid](self, output, n, BLOCK=1024)
    return output
