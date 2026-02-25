"""
Torch参考实现 - addcmul
算子类型: general
描述: The addcmul operator
"""

import torch

def addcmul(inp, tensor1, tensor2, *, value=1.0, out=None):
    return torch.addcmul(inp, tensor1, tensor2, value, out)