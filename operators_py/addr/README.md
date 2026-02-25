# addr

## 基本信息

- **算子名**: addr
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The addr operator

## 查询语句

操作符名字是 addr，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| input | torch.Tensor | Parameter input |
| vec1 | Any | Parameter vec1 |
| vec2 | Any | Parameter vec2 |
| beta | float | Parameter beta |
| alpha | float | Parameter alpha |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `addr_triton.py` - Triton kernel实现（FlagGems原始代码）
- `addr_torch.py` - PyTorch参考实现（groundtruth）
- `addr_test.py` - 测试代码（bench格式）
