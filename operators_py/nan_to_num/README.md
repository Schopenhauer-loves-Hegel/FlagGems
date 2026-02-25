# nan_to_num

## 基本信息

- **算子名**: nan_to_num
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The nan_to_num operator

## 查询语句

操作符名字是 nan_to_num，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| A | Any | Parameter A |
| nan | Any | Parameter nan |
| posinf | Any | Parameter posinf |
| neginf | Any | Parameter neginf |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `nan_to_num_triton.py` - Triton kernel实现（FlagGems原始代码）
- `nan_to_num_torch.py` - PyTorch参考实现（groundtruth）
- `nan_to_num_test.py` - 测试代码（bench格式）
