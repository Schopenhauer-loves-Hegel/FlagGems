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


class EinsumBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_SHAPES = [(1, 512, 512, 512), (1, 1024, 1024, 1024), (16, 512, 512, 512)]

    def __init__(self, *args, batched=False, equation=None, input_fn=None, **kwargs):
        self.batched = batched
        self.equation = equation
        super().__init__(*args, **kwargs)

    def set_more_shapes(self):
        return []

    def set_shapes(self, *args, **kwargs):
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for case in self.get_case_iter(dtype):
            yield self.build_inputs(case)

    def supports_cases(self) -> bool:
        return type(self).get_input_iter is EinsumBenchmark.get_input_iter

    def get_case_iter(self, dtype) -> Generator:
        for ordinal, (b, m, n, k) in enumerate(self.shapes):
            input_shapes = (
                ((b, m, k), (b, k, n))
                if self.batched
                else ((m, k), (k, n))
            )
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"inputs": input_shapes},
                    params={"equation": self.equation},
                    builder_args=input_shapes,
                ),
            )

    def build_inputs(self, case):
        shape_a, shape_b = case.builder_args[0].builder_args
        inp1 = torch.randn(shape_a, dtype=case.dtype, device=self.device)
        inp2 = torch.randn(shape_b, dtype=case.dtype, device=self.device)
        return inp1, inp2

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        if self.batched:
            return A.shape[0] * A.shape[1] * B.shape[2] * A.shape[2] * 2
        return A.shape[0] * B.shape[1] * A.shape[1] * 2


class EinsumGenericBenchmark(base.GenericBenchmark):
    def set_shapes(self, *args, **kwargs):
        pass  # keep shapes set by caller


def dot_input_fn(shape, dtype, device):
    (n,) = shape
    yield torch.randn(n, dtype=dtype, device=device), torch.randn(
        n, dtype=dtype, device=device
    )


def outer_input_fn(shape, dtype, device):
    m, n = shape
    yield torch.randn(m, dtype=dtype, device=device), torch.randn(
        n, dtype=dtype, device=device
    )


def unary_2d_input_fn(shape, dtype, device):
    m, n = shape
    yield (torch.randn(m, n, dtype=dtype, device=device),)


def unary_3d_input_fn(shape, dtype, device):
    m, n, k = shape
    yield (torch.randn(m, n, k, dtype=dtype, device=device),)


def ellipsis_input_fn(shape, dtype, device):
    b, h, m, k, n = shape
    yield torch.randn(b, h, m, k, dtype=dtype, device=device), torch.randn(
        b, h, k, n, dtype=dtype, device=device
    )


def _case_fn_factory(input_shapes_fn, equation):
    def inner(shape, dtype):
        del dtype
        input_shapes = input_shapes_fn(shape)
        yield base.BenchmarkCasePlan(
            shape={"inputs": input_shapes},
            params={"equation": equation},
            builder_args=(shape, 0),
        )

    return inner


@pytest.mark.einsum
def test_einsum_matmul():
    bench = EinsumBenchmark(
        input_fn=None,
        equation="ij,jk->ik",
        op_name="einsum",
        torch_op=lambda A, B: torch.einsum("ij,jk->ik", A, B),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.einsum
def test_einsum_bmm():
    bench = EinsumBenchmark(
        input_fn=None,
        equation="bij,bjk->bik",
        op_name="einsum",
        torch_op=lambda A, B: torch.einsum("bij,bjk->bik", A, B),
        dtypes=consts.FLOAT_DTYPES,
        batched=True,
    )
    bench.run()


@pytest.mark.einsum
def test_einsum_dot():
    bench = EinsumGenericBenchmark(
        case_fn=_case_fn_factory(lambda shape: [shape, shape], "i,i->"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(dot_input_fn),
        op_name="einsum",
        torch_op=lambda A, B: torch.einsum("i,i->", A, B),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024,), (4096,), (65536,)]
    bench.run()


@pytest.mark.einsum
def test_einsum_outer():
    bench = EinsumGenericBenchmark(
        case_fn=_case_fn_factory(
            lambda shape: [(shape[0],), (shape[1],)], "i,j->ij"
        ),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(outer_input_fn),
        op_name="einsum",
        torch_op=lambda A, B: torch.einsum("i,j->ij", A, B),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_trace():
    bench = EinsumGenericBenchmark(
        case_fn=_case_fn_factory(lambda shape: [shape], "ii->"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(unary_2d_input_fn),
        op_name="einsum",
        torch_op=lambda A: torch.einsum("ii->", A),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_diagonal():
    bench = EinsumGenericBenchmark(
        case_fn=_case_fn_factory(lambda shape: [shape], "ii->i"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(unary_2d_input_fn),
        op_name="einsum",
        torch_op=lambda A: torch.einsum("ii->i", A),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_transpose():
    bench = EinsumGenericBenchmark(
        case_fn=_case_fn_factory(lambda shape: [shape], "ij->ji"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(unary_2d_input_fn),
        op_name="einsum",
        torch_op=lambda A: torch.einsum("ij->ji", A),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(1024, 1024), (4096, 4096)]
    bench.run()


@pytest.mark.einsum
def test_einsum_sum_all():
    bench = EinsumGenericBenchmark(
        case_fn=_case_fn_factory(lambda shape: [shape], "ijk->"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(unary_3d_input_fn),
        op_name="einsum",
        torch_op=lambda A: torch.einsum("ijk->", A),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(64, 64, 64), (128, 128, 128)]
    bench.run()


@pytest.mark.einsum
def test_einsum_sum_dim():
    bench = EinsumGenericBenchmark(
        case_fn=_case_fn_factory(lambda shape: [shape], "ijk->j"),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(unary_3d_input_fn),
        op_name="einsum",
        torch_op=lambda A: torch.einsum("ijk->j", A),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(64, 64, 64), (128, 128, 128)]
    bench.run()


@pytest.mark.einsum
def test_einsum_ellipsis():
    bench = EinsumGenericBenchmark(
        case_fn=_case_fn_factory(
            lambda shape: [
                (shape[0], shape[1], shape[2], shape[3]),
                (shape[0], shape[1], shape[3], shape[4]),
            ],
            "...ij,...jk->...ik",
        ),
        build_inputs_fn=base.build_inputs_from_generic_input_fn(ellipsis_input_fn),
        op_name="einsum",
        torch_op=lambda A, B: torch.einsum("...ij,...jk->...ik", A, B),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = [(2, 4, 64, 64, 128), (2, 8, 128, 128, 256)]
    bench.run()
