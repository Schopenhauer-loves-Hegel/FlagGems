# outer

## 基本信息

- **算子名**: outer
- **算子类型**: blas
- **目标硬件**: nvidia
- **描述**: The outer operator

## 查询语句

操作符名字是 outer，是一个 blas 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| weight | torch.Tensor | Parameter weight |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `outer_triton.py` - Triton kernel实现（FlagGems原始代码）
- `outer_torch.py` - PyTorch参考实现（groundtruth）
- `outer_test.py` - 测试代码（bench格式）
