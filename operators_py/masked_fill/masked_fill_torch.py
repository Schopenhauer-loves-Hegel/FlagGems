"""
Torch参考实现 - masked_fill
算子类型: general
描述: The masked_fill operator
"""

import torch

def masked_fill(inp, mask, value):
    return torch.masked_fill(inp, mask, value)