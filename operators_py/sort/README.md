# sort

## 基本信息

- **算子名**: sort
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The sort operator

## 查询语句

操作符名字是 sort，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| dim | int | Parameter dim |
| descending | bool | Parameter descending |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `sort_triton.py` - Triton kernel实现（FlagGems原始代码）
- `sort_torch.py` - PyTorch参考实现（groundtruth）
- `sort_test.py` - 测试代码（bench格式）
