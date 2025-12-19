"""
Tests for experimental operator: huber_loss

Auto-imported from generated test code.
Op ID: 60a39af4-7b4d-4d9b-b511-d8915b1e5e76
"""

import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import to_reference
from sandbox.register import REGISTERED_OPS
import torch

@label("huber_loss")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
@parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_tensor(shape, dtype, reduction, delta):
    self_tensor = torch.randn(shape, dtype=dtype, device="cuda")
    target_tensor = torch.randn(shape, dtype=dtype, device="cuda")

    ref_self = self_tensor.clone()
    ref_target = target_tensor.clone()
    ref_out = torch.ops.aten.huber_loss(ref_self, ref_target, reduction, float(delta))

    with flagbench.use_gems(REGISTERED_OPS):
        act_self = self_tensor.clone()
        act_target = target_tensor.clone()
        act_out = torch.ops.aten.huber_loss(act_self, act_target, reduction, float(delta))

    assert_close(act_out, ref_out, dtype=dtype)


@label("huber_loss")
@parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@parametrize("reduction", [0, 1, 2])
@parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_out(shape, dtype, reduction, delta):
    self_tensor = torch.randn(shape, dtype=dtype, device="cuda")
    target_tensor = torch.randn(shape, dtype=dtype, device="cuda")

    if reduction == 0:
        out_shape = shape
    else:
        out_shape = ()

    ref_self = self_tensor.clone()
    ref_target = target_tensor.clone()
    ref_out = torch.empty(out_shape, dtype=dtype, device="cuda")
    torch.ops.aten.huber_loss.out(ref_self, ref_target, reduction, float(delta), out=ref_out)

    with flagbench.use_gems(REGISTERED_OPS):
        act_self = self_tensor.clone()
        act_target = target_tensor.clone()
        act_out = torch.empty(out_shape, dtype=dtype, device="cuda")
        torch.ops.aten.huber_loss.out(act_self, act_target, reduction, float(delta), out=act_out)

    assert_close(act_out, ref_out, dtype=dtype)
