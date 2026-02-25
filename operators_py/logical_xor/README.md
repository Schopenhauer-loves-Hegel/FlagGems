# logical_xor

## 基本信息

- **算子名**: logical_xor
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The logical_xor operator

## 查询语句

操作符名字是 logical_xor，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| A | Any | Parameter A |
| B | Any | Parameter B |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `logical_xor_triton.py` - Triton kernel实现（FlagGems原始代码）
- `logical_xor_torch.py` - PyTorch参考实现（groundtruth）
- `logical_xor_test.py` - 测试代码（bench格式）
