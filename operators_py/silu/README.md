# silu

## 基本信息

- **算子名**: silu
- **算子类型**: pointwise
- **目标硬件**: nvidia
- **描述**: The silu operator

## 查询语句

操作符名字是 silu，是一个 pointwise 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| self | Any | Parameter self |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `silu_triton.py` - Triton kernel实现（FlagGems原始代码）
- `silu_torch.py` - PyTorch参考实现（groundtruth）
- `silu_test.py` - 测试代码（bench格式）
