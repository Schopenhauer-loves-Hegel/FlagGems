"""
Torch参考实现 - where
算子类型: pointwise
描述: The where operator
"""

import torch

def where(input):
    return torch.where(condition, x, y)