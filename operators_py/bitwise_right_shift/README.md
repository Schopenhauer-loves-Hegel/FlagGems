# bitwise_right_shift

## 基本信息

- **算子名**: bitwise_right_shift
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The bitwise_right_shift operator

## 查询语句

操作符名字是 bitwise_right_shift，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| self | Any | Parameter self |
| other | torch.Tensor | Parameter other |
| out | Any | Parameter out |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `bitwise_right_shift_triton.py` - Triton kernel实现（FlagGems原始代码）
- `bitwise_right_shift_torch.py` - PyTorch参考实现（groundtruth）
- `bitwise_right_shift_test.py` - 测试代码（bench格式）
