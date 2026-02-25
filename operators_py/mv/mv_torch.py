"""
Torch参考实现 - mv
算子类型: blas
描述: The mv operator
"""

import torch

def mv(inp, vec):
    return torch.mv(inp, vec)