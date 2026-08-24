# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from . import base, consts


@pytest.mark.is_nonzero
def test_is_nonzero():
    def is_nonzero_case_fn(shape, dtype):
        del shape, dtype
        yield base.BenchmarkCasePlan(
            shape={"input": (1,)},
            params={"value": 1},
        )

    def is_nonzero_build_inputs_fn(plan, dtype, device):
        del plan
        return (torch.tensor([1], dtype=dtype, device=device),)

    bench = base.GenericBenchmark(
        case_fn=is_nonzero_case_fn,
        build_inputs_fn=is_nonzero_build_inputs_fn,
        op_name="is_nonzero",
        torch_op=torch.is_nonzero,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(lambda x: torch.is_nonzero(x))
    bench.run()
