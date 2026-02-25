"""
Torch参考实现 - stack
算子类型: tensor_ops
描述: The stack operator
"""

import torch

def stack(*args, **kwargs):
    return torch.stack(tensors, dim=dim)