# vdot

## 基本信息

- **算子名**: vdot
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The vdot operator

## 查询语句

操作符名字是 vdot，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| input: Tensor | torch.Tensor | Parameter input: Tensor |
| other: Tensor | torch.Tensor | Parameter other: Tensor |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `vdot_triton.py` - Triton kernel实现（FlagGems原始代码）
- `vdot_torch.py` - PyTorch参考实现（groundtruth）
- `vdot_test.py` - 测试代码（bench格式）
