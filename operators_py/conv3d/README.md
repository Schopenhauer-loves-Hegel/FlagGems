# conv3d

## 基本信息

- **算子名**: conv3d
- **算子类型**: general
- **目标硬件**: nvidia
- **描述**: The conv3d operator

## 查询语句

操作符名字是 conv3d，是一个 general 算子，处理硬件是 Nvidia。

## 输入参数

| 参数名 | 类型 | 描述 |
|--------|------|------|
| input | torch.Tensor | Parameter input |
| weight | torch.Tensor | Parameter weight |
| bias | torch.Tensor | Parameter bias |
| stride | int | Parameter stride |
| padding | int | Parameter padding |
| dilation | int | Parameter dilation |
| groups | int | Parameter groups |

## 输出参数

| 类型 | 描述 |
|------|------|
| torch.Tensor | The output tensor |

## 文件说明

- `conv3d_triton.py` - Triton kernel实现（FlagGems原始代码）
- `conv3d_torch.py` - PyTorch参考实现（groundtruth）
- `conv3d_test.py` - 测试代码（bench格式）
