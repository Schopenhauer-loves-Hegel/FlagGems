"""
测试代码 - exp2
算子类型: general
"""

import bench
from bench.sandbox.test.test_parametrize import parametrize, label
from bench.sandbox.config import DEVICE as device
from bench.sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from bench.sandbox.utils.accuracy_utils import to_reference
import torch

@label("exp2")
@parametrize("shape", [(32, 32), (64, 64), (128, 128)])
@parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_exp2(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = to_reference(x, True)

    ref_out = bench.exp2(ref_x)
    res_out = bench.triton.exp2(x)

    assert_close(res_out, ref_out, dtype)