"""
Torch参考实现 - threshold
算子类型: general
描述: The threshold operator
"""

import torch

def threshold(self, threshold, value):
    return torch.threshold(self, threshold, value)