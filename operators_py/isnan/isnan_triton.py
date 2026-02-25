"""
Triton实现 - isnan
算子类型: pointwise
描述: The isnan operator
"""

import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic, tl_extra_shim

_isnan = tl_extra_shim.isnan

logger = logging.getLogger(__name__)

@pointwise_dynamic(promotion_methods=[(0, "ALWAYS_BOOL")])
@triton.jit
def isnan_func(x):
    return _isnan(x.to(tl.float32))

def isnan(A):
    logger.debug("GEMS ISNAN")
    return isnan_func(A)