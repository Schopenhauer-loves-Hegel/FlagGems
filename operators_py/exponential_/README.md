# exponential_

## 基本信息

- **算子名**: exponential_
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The exponential_ operator

## 查询语句

操作符名字是 exponential_，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | torch.Tensor | Parameter x |
| lambd: float | float | Parameter lambd: float |
| generator | Any | Parameter generator |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `exponential__triton.py` - Triton kernel实现（FlagGems原始代码）
- `exponential__torch.py` - PyTorch参考实现（groundtruth）
- `exponential__test.py` - 测试代码（bench格式）
