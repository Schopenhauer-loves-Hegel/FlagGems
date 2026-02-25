# conv_depthwise2d

## 基本信息

- **算子名**: conv_depthwise2d
- **算子类型**: conv
- **目标硬件**: nvidia
- **描述**: The conv_depthwise2d operator

## 查询语句

操作符名字是 conv_depthwise2d，是一个 conv 算子，处理硬件是 Nvidia。

## 输入参数

无参数信息

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `conv_depthwise2d_triton.py` - Triton kernel实现（FlagGems原始代码）
- `conv_depthwise2d_torch.py` - PyTorch参考实现（groundtruth）
- `conv_depthwise2d_test.py` - 测试代码（bench格式）
