# mse_loss

## 基本信息

- **算子名**: mse_loss
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The mse_loss operator

## 查询语句

操作符名字是 mse_loss，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| target | Any | Parameter target |
| reduction | float | Parameter reduction |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `mse_loss_triton.py` - Triton kernel实现（FlagGems原始代码）
- `mse_loss_torch.py` - PyTorch参考实现（groundtruth）
- `mse_loss_test.py` - 测试代码（bench格式）
