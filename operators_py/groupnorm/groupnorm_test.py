"""
测试代码 - groupnorm
算子类型: general
"""

import bench
from bench.sandbox.test.test_parametrize import parametrize, label
from bench.sandbox.config import DEVICE as device
from bench.sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from bench.sandbox.utils.accuracy_utils import to_reference
import torch

@label("groupnorm")
def test_groupnorm(N, C, H, W, num_groups, dtype, wb_none):
    # TODO: Implement test body
    pass