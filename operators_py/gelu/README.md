# gelu

## 基本信息

- **算子名**: gelu
- **算子类型**: pointwise
- **目标硬件**: nvidia
- **描述**: The gelu operator

## 查询语句

操作符名字是 gelu，是一个 pointwise 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| self | Any | Parameter self |
| approximate | torch.Tensor | Parameter approximate |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `gelu_triton.py` - Triton kernel实现（FlagGems原始代码）
- `gelu_torch.py` - PyTorch参考实现（groundtruth）
- `gelu_test.py` - 测试代码（bench格式）
