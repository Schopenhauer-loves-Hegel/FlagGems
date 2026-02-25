"""
Torch参考实现 - clamp
算子类型: pointwise
描述: The clamp operator
"""

import torch

def clamp(A, mini=None, maxi=None):
    return torch.clamp(A, mini)