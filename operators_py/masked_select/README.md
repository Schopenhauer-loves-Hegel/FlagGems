# masked_select

## 基本信息

- **算子名**: masked_select
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The masked_select operator

## 查询语句

操作符名字是 masked_select，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| mask | Any | Parameter mask |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `masked_select_triton.py` - Triton kernel实现（FlagGems原始代码）
- `masked_select_torch.py` - PyTorch参考实现（groundtruth）
- `masked_select_test.py` - 测试代码（bench格式）
