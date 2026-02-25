# scaled_softmax

## 基本信息

- **算子名**: scaled_softmax
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The scaled_softmax operator

## 查询语句

操作符名字是 scaled_softmax，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `scaled_softmax_triton.py` - Triton kernel实现（FlagGems原始代码）
- `scaled_softmax_torch.py` - PyTorch参考实现（groundtruth）
- `scaled_softmax_test.py` - 测试代码（bench格式）
