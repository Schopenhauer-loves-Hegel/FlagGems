"""
Accuracy tests for top 7 most frequent matmul shapes.
Based on profiling data from matmul_shape.csv.
"""
import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    gems_assert_close,
    to_reference,
)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mm_shape1_4x1024_1024x2048(dtype):
    """Test shape: (4, 1024) @ (1024, 2048) - count: 2688"""
    M, K, N = 4, 1024, 2048

    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")

    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mm_shape2_4x2048_2048x256(dtype):
    """Test shape: (4, 2048) @ (2048, 256) - count: 2688"""
    M, K, N = 4, 2048, 256

    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")

    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)

@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mm_shape3_4x128_128x2048(dtype):
    """Test shape: (4, 128) @ (128, 2048) - count: 2688"""
    M, K, N = 4, 128, 2048

    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")

    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mm_shape4_4x2048_2048x1(dtype):
    """Test shape: (4, 2048) @ (2048, 1) - count: 2688"""
    M, K, N = 4, 2048, 1

    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")

    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mm_shape5_4x2048_2048x512(dtype):
    """Test shape: (4, 2048) @ (2048, 512) - count: 2688"""
    M, K, N = 4, 2048, 512

    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")

    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mm_shape6_4x2048_2048x3072(dtype):
    """Test shape: (4, 2048) @ (2048, 3072) - count: 2016"""
    M, K, N = 4, 2048, 3072

    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")

    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)


@pytest.mark.mm
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_mm_shape7_4x2048_2048x16(dtype):
    """Test shape: (4, 2048) @ (2048, 16) - count: 2016"""
    M, K, N = 4, 2048, 16

    mat1 = torch.randn((M, K), dtype=dtype, device="cuda")
    mat2 = torch.randn((K, N), dtype=dtype, device="cuda")

    ref_mat1 = to_reference(mat1, True)
    ref_mat2 = to_reference(mat2, True)

    ref_out = torch.mm(ref_mat1, ref_mat2)
    with flag_gems.use_gems():
        res_out = torch.mm(mat1, mat2)

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=K)
