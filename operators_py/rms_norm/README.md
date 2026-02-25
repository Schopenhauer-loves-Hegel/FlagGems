# rms_norm

## 基本信息

- **算子名**: rms_norm
- **算子类型**: normalization
- **目标硬件**: nvidia
- **描述**: The rms_norm operator

## 查询语句

操作符名字是 rms_norm，是一个 normalization 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | torch.Tensor | Parameter x |
| normalized_shape | Any | Parameter normalized_shape |
| weight | torch.Tensor | Parameter weight |
| eps | float | Parameter eps |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `rms_norm_triton.py` - Triton kernel实现（FlagGems原始代码）
- `rms_norm_torch.py` - PyTorch参考实现（groundtruth）
- `rms_norm_test.py` - 测试代码（bench格式）
