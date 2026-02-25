"""
Torch参考实现 - elu
算子类型: general
描述: The elu operator
"""

import torch

def elu(A, alpha=1.0, scale=1.0, input_scale=1.0):
    return torch.elu(A, alpha, scale, input_scale)