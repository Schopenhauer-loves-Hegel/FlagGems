# baddbmm

## 基本信息

- **算子名**: baddbmm
- **算子类型**: blas
- **目标硬件**: nvidia
- **描述**: The baddbmm operator

## 查询语句

操作符名字是 baddbmm，是一个 blas 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| bias | torch.Tensor | Parameter bias |
| A | Any | Parameter A |
| B | Any | Parameter B |
| beta | float | Parameter beta |
| alpha | float | Parameter alpha |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `baddbmm_triton.py` - Triton kernel实现（FlagGems原始代码）
- `baddbmm_torch.py` - PyTorch参考实现（groundtruth）
- `baddbmm_test.py` - 测试代码（bench格式）
