"""
测试代码 - weightnorm
算子类型: general
"""

import bench
from bench.sandbox.test.test_parametrize import parametrize, label
from bench.sandbox.config import DEVICE as device
from bench.sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from bench.sandbox.utils.accuracy_utils import to_reference
import torch

@label("weightnorm")
def test_weightnorm(shape, dtype, dim):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = to_reference(x, True)

    ref_out = bench.weightnorm(ref_x)
    res_out = bench.triton.weightnorm(x)

    assert_close(res_out, ref_out, dtype)