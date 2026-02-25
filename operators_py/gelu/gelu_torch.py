"""
Torch参考实现 - gelu
算子类型: pointwise
描述: The gelu operator
"""

import torch

def gelu(self, *, approximate="none"):
    return torch.gelu(self, approximate)