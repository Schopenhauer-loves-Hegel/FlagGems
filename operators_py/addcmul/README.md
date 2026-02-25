# addcmul

## 基本信息

- **算子名**: addcmul
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The addcmul operator

## 查询语句

操作符名字是 addcmul，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| tensor1 | torch.Tensor | Parameter tensor1 |
| tensor2 | torch.Tensor | Parameter tensor2 |
| value | float | Parameter value |
| out | Any | Parameter out |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `addcmul_triton.py` - Triton kernel实现（FlagGems原始代码）
- `addcmul_torch.py` - PyTorch参考实现（groundtruth）
- `addcmul_test.py` - 测试代码（bench格式）
