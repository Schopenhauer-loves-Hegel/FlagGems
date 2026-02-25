"""
Torch参考实现 - masked_select
算子类型: general
描述: The masked_select operator
"""

import torch

def masked_select(inp, mask):
    return torch.masked_select(inp, mask)