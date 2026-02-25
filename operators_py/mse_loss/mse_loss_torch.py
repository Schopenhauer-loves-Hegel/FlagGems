"""
Torch参考实现 - mse_loss
算子类型: general
描述: The mse_loss operator
"""

import torch

def mse_loss(inp, target, reduction=Reduction.MEAN.value):
    return torch.mse_loss(inp, target, reduction)