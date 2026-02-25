# exp2

## 基本信息

- **算子名**: exp2
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The exp2 operator

## 查询语句

操作符名字是 exp2，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| A | Any | Parameter A |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `exp2_triton.py` - Triton kernel实现（FlagGems原始代码）
- `exp2_torch.py` - PyTorch参考实现（groundtruth）
- `exp2_test.py` - 测试代码（bench格式）
