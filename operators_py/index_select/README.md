# index_select

## 基本信息

- **算子名**: index_select
- **算子类型**: indexing
- **目标硬件**: nvidia
- **描述**: The index_select operator

## 查询语句

操作符名字是 index_select，是一个 indexing 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| inp | Any | Parameter inp |
| dim | int | Parameter dim |
| index | torch.Tensor | Parameter index |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `index_select_triton.py` - Triton kernel实现（FlagGems原始代码）
- `index_select_torch.py` - PyTorch参考实现（groundtruth）
- `index_select_test.py` - 测试代码（bench格式）
