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

from typing import Generator

import pytest
import torch

from . import base, consts


class TensorSplitBenchmark(base.Benchmark):
    """Benchmark for tensor_split operator."""

    def set_shapes(self, shape_file_path=None):
        # Various shapes covering 1D to 4D tensors for split benchmarking
        self.shapes = [
            (64,),
            (256,),
            (1024,),
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (8, 16, 32),
            (16, 32, 64),
            (32, 64, 128),
            (4, 8, 16, 32),
            (8, 16, 32, 64),
        ]

    def get_input_iter(self, cur_dtype) -> Generator:
        for case in self.get_case_iter(cur_dtype):
            yield self.build_inputs(case)

    def get_case_iter(self, cur_dtype) -> Generator:
        for ordinal, shape in enumerate(self.shapes):
            # Split into 3 sections
            sections = 3
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": shape},
                    params={"indices_or_sections": sections, "dim": 0},
                    builder_args=(shape, sections),
                ),
            )

    def build_inputs(self, case):
        shape, sections = case.builder_args[0].builder_args
        inp = base.generate_tensor_input(shape, case.dtype, self.device)
        return inp, sections


@pytest.mark.tensor_split
def test_tensor_split():
    bench = TensorSplitBenchmark(
        op_name="tensor_split",
        torch_op=torch.tensor_split,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
