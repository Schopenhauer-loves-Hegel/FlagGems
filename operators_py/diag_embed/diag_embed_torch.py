"""
Torch参考实现 - diag_embed
算子类型: general
描述: The diag_embed operator
"""

import torch

def diag_embed(x, offset=0, dim1=-2, dim2=-1):
    return torch.diag_embed(x, offset, dim1, dim2)