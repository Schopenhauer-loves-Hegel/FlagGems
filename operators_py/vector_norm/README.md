# vector_norm

## 基本信息

- **算子名**: vector_norm
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The vector_norm operator

## 查询语句

操作符名字是 vector_norm，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| x | torch.Tensor | Parameter x |
| ord | int | Parameter ord |
| dim | int | Parameter dim |
| keepdim | int | Parameter keepdim |
| dtype | torch.Tensor | Parameter dtype |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `vector_norm_triton.py` - Triton kernel实现（FlagGems原始代码）
- `vector_norm_torch.py` - PyTorch参考实现（groundtruth）
- `vector_norm_test.py` - 测试代码（bench格式）
