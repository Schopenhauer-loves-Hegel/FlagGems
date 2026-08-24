import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_sinc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_sinc(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.sinc(ref_inp)
    gems_op = flag_gems.testing.resolve_gems_op("special_sinc", flag_gems.special_sinc)
    res_out = gems_op(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
