"""
测试代码 - vdot
算子类型: general
"""

import bench
from bench.sandbox.test.test_parametrize import parametrize, label
from bench.sandbox.config import DEVICE as device
from bench.sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from bench.sandbox.utils.accuracy_utils import to_reference
import torch

@label("vdot")
@parametrize("shape", [(32, 32), (64, 64), (128, 128)])
@parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_vdot(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = to_reference(x, True)

    ref_out = bench.vdot(ref_x)
    res_out = bench.triton.vdot(x)

    assert_close(res_out, ref_out, dtype)