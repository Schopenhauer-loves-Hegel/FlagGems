# var_mean

## 基本信息

- **算子名**: var_mean
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The var_mean operator

## 查询语句

操作符名字是 var_mean，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | torch.Tensor | Parameter x |
| dim | int | Parameter dim |
| correction | Any | Parameter correction |
| keepdim | int | Parameter keepdim |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `var_mean_triton.py` - Triton kernel实现（FlagGems原始代码）
- `var_mean_torch.py` - PyTorch参考实现（groundtruth）
- `var_mean_test.py` - 测试代码（bench格式）
