"""
Torch参考实现 - softplus
算子类型: general
描述: The softplus operator
"""

import torch

def softplus(self, beta=1.0, threshold=20.0):
    return torch.softplus(self, beta, threshold)