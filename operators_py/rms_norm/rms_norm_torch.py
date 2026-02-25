"""
Torch参考实现 - rms_norm
算子类型: normalization
描述: The rms_norm operator
"""

import torch

def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return torch.rms_norm(x, normalized_shape, weight, eps)