# eye_m

## 基本信息

- **算子名**: eye_m
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The eye_m operator

## 查询语句

操作符名字是 eye_m，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| n | Any | Parameter n |
| m | Any | Parameter m |
| dtype | torch.Tensor | Parameter dtype |
| layout | torch.Tensor | Parameter layout |
| device | Any | Parameter device |
| pin_memory | torch.Tensor | Parameter pin_memory |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `eye_m_triton.py` - Triton kernel实现（FlagGems原始代码）
- `eye_m_torch.py` - PyTorch参考实现（groundtruth）
- `eye_m_test.py` - 测试代码（bench格式）
