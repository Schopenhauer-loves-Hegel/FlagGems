# mm

## 基本信息

- **算子名**: mm
- **算子类型**: blas
- **目标硬件**: nvidia
- **描述**: The mm operator

## 查询语句

操作符名字是 mm，是一个 blas 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| a | Any | Parameter a |
| b | Any | Parameter b |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `mm_triton.py` - Triton kernel实现（FlagGems原始代码）
- `mm_torch.py` - PyTorch参考实现（groundtruth）
- `mm_test.py` - 测试代码（bench格式）
