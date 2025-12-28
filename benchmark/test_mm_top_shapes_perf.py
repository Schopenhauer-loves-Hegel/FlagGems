"""
Performance tests for top 7 most frequent matmul shapes.
Based on profiling data from matmul_shape.csv.
"""
import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark


class TopShapeBenchmark(Benchmark):
    """
    Benchmark for specific top frequency shapes
    """

    DEFAULT_METRICS = ["latency_base", "latency", "tflops", "speedup"]

    def __init__(self, *args, shape, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = shape

    def get_input_iter(self, cur_dtype):
        M, K, N = self.shape
        mat1 = torch.randn([M, K], dtype=cur_dtype, device=self.device)
        mat2 = torch.randn([K, N], dtype=cur_dtype, device=self.device)
        yield mat1, mat2

    def get_tflops(self, op, *args, **kwargs):
        # shape(m,k)(k,n)
        # total_flops = m * n * 2k
        M = args[0].shape[0]
        K = args[0].shape[1]
        N = args[1].shape[1]
        total_flops = M * N * 2 * K
        return total_flops


@pytest.mark.mm
def test_mm_shape1_4x1024_1024x2048_perf():
    """Test shape: (4, 1024) @ (1024, 2048) - count: 2688"""
    shape = (4, 1024, 2048)
    bench = TopShapeBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16],
        shape=shape,
    )
    bench.run()


@pytest.mark.mm
def test_mm_shape2_4x2048_2048x256_perf():
    """Test shape: (4, 2048) @ (2048, 256) - count: 2688"""
    shape = (4, 2048, 256)
    bench = TopShapeBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16],
        shape=shape,
    )
    bench.run()


@pytest.mark.mm
def test_mm_shape3_4x128_128x2048_perf():
    """Test shape: (4, 128) @ (128, 2048) - count: 2688"""
    shape = (4, 128, 2048)
    bench = TopShapeBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16],
        shape=shape,
    )
    bench.run()


@pytest.mark.mm
def test_mm_shape4_4x2048_2048x1_perf():
    """Test shape: (4, 2048) @ (2048, 1) - count: 2688"""
    shape = (4, 2048, 1)
    bench = TopShapeBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16],
        shape=shape,
    )
    bench.run()


@pytest.mark.mm
def test_mm_shape5_4x2048_2048x512_perf():
    """Test shape: (4, 2048) @ (2048, 512) - count: 2688"""
    shape = (4, 2048, 512)
    bench = TopShapeBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16],
        shape=shape,
    )
    bench.run()


@pytest.mark.mm
def test_mm_shape6_4x2048_2048x3072_perf():
    """Test shape: (4, 2048) @ (2048, 3072) - count: 2016"""
    shape = (4, 2048, 3072)
    bench = TopShapeBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16],
        shape=shape,
    )
    bench.run()


@pytest.mark.mm
def test_mm_shape7_4x2048_2048x16_perf():
    """Test shape: (4, 2048) @ (2048, 16) - count: 2016"""
    shape = (4, 2048, 16)
    bench = TopShapeBenchmark(
        op_name="mm",
        torch_op=torch.mm,
        dtypes=[torch.bfloat16],
        shape=shape,
    )
    bench.run()
