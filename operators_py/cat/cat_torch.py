"""
Torch参考实现 - cat
算子类型: tensor_ops
描述: The cat operator
"""

import torch

def cat(*args, **kwargs):
    return torch.cat(tensors, dim=dim)