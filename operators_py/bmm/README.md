# bmm

## 基本信息

- **算子名**: bmm
- **算子类型**: blas
- **目标硬件**: nvidia
- **描述**: The bmm operator

## 查询语句

操作符名字是 bmm，是一个 blas 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| A | Any | Parameter A |
| B | Any | Parameter B |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `bmm_triton.py` - Triton kernel实现（FlagGems原始代码）
- `bmm_torch.py` - PyTorch参考实现（groundtruth）
- `bmm_test.py` - 测试代码（bench格式）
