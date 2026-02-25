"""
Triton实现 - bitwise_left_shift
算子类型: general
描述: The bitwise_left_shift operator
"""

import logging

import triton

from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_left_shift_kernel(a, b):
    return a << b

def bitwise_left_shift(self, other, *, out=None):
    logger.debug("GEMS BITWISE_LEFT_SHIFT")
    return bitwise_left_shift_kernel(self, other, out=out)