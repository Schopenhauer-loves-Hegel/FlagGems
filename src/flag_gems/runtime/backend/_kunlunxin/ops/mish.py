import logging

import triton
import triton.language as tl

from flag_gems.utils import tl_extra_shim

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)
_tanh = tl_extra_shim.tanh


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def mish_forward(x):
    x_fp32 = x.to(tl.float32)
    # mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    sp = tl.log(1.0 + tl.exp(x_fp32))
    out = x_fp32 * _tanh(sp)
    return out


def mish(self):
    logger.debug("GEMS_KUNLUNXIN MISH_FORWARD")
    output = mish_forward(self)
    return output
