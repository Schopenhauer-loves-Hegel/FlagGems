"""
Torch参考实现 - log_softmax
算子类型: normalization
描述: The log_softmax operator
"""

import torch

def log_softmax(self, dim, half_to_float=False):
    return torch.log_softmax(self, dim=dim)