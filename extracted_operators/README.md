# FlagGems 算子数据集

本目录包含从 [FlagGems](https://github.com/FlagOpen/FlagGems) 仓库提取的 **130个算子** 的完整数据。

## 📊 数据集概览

- **算子总数**: 130 个
- **数据格式**: JSON
- **组织方式**: 每个算子一个独立的 JSON 文件
- **总大小**: 718KB

## 📋 算子类型分布

| 算子类型 | 数量 | 说明 |
|---------|------|------|
| general | 70 | 通用算子 |
| pointwise | 30 | 逐点操作算子（add, mul, relu等） |
| reduction | 11 | 规约操作算子（sum, mean, max等） |
| blas | 5 | 基础线性代数算子（mm, bmm等） |
| indexing | 4 | 索引操作算子（gather, scatter等） |
| normalization | 4 | 归一化算子（softmax, layer_norm等） |
| conv | 3 | 卷积算子 |
| tensor_ops | 2 | 张量操作算子 |
| attention | 1 | 注意力机制算子 |

## 📦 数据格式

每个 JSON 文件包含以下字段：

```json
{
    "query": "操作符名字是 xxx，是一个 xxx 算子，处理硬件是 Nvidia。",
    "kernel_name": "算子名称",
    "func_desc": "算子功能描述（英文）",
    "func_type": "算子类型（pointwise/reduction/blas等）",
    "gpu": "nvidia",
    "input_args": [
        {
            "name": "参数名",
            "type": "参数类型",
            "desc": "参数描述"
        }
    ],
    "output_args": [
        {
            "type": "torch.Tensor",
            "desc": "输出描述"
        }
    ],
    "triton_kernel_code": "完整的Triton实现代码",
    "torch_kernel_code": "纯Torch参考实现代码",
    "test_func_code": "测试函数代码（bench格式）"
}
```

## 🎯 三种实现

### 1. Triton Kernel Code
- **来源**: FlagGems 原始实现
- **特点**:
  - 包含完整的 Triton kernel 代码
  - 使用 `@triton.jit` 装饰器
  - 部分使用 `@pointwise_dynamic` 自动生成
  - 包含前向和反向传播实现

### 2. Torch Kernel Code
- **来源**: 根据算子语义生成
- **特点**:
  - 纯 PyTorch API 实现
  - 作为正确性参考（groundtruth）
  - 参数名与原始实现保持一致
  - 可直接运行验证

### 3. Test Func Code
- **来源**: 从 FlagGems 测试代码转换
- **格式**: bench 库风格
- **特点**:
  - 使用 `@parametrize` 装饰器
  - 包含多种测试配置
  - 对比 triton 和 torch 实现的精度

## 📚 示例

### add.json

```json
{
    "kernel_name": "add",
    "func_type": "pointwise",
    "input_args": [
        {"name": "A", "type": "Any"},
        {"name": "B", "type": "Any"},
        {"name": "alpha", "type": "float"}
    ],
    "torch_kernel_code": "import torch\n\ndef add(A, B, *, alpha=1):\n    return torch.add(A, B, alpha=alpha)"
}
```

### gather.json

```json
{
    "kernel_name": "gather",
    "func_type": "indexing",
    "input_args": [
        {"name": "inp", "type": "Any"},
        {"name": "dim", "type": "int"},
        {"name": "index", "type": "torch.Tensor"}
    ],
    "torch_kernel_code": "import torch\n\ndef gather(inp, dim, index, out=None, sparse_grad=False):\n    return torch.gather(inp, dim, index)"
}
```

## 🔧 使用方法

### 读取单个算子

```python
import json

# 读取 add 算子
with open('extracted_operators/add.json', 'r', encoding='utf-8') as f:
    add_op = json.load(f)

print(f"算子名: {add_op['kernel_name']}")
print(f"类型: {add_op['func_type']}")
print(f"Torch实现:\n{add_op['torch_kernel_code']}")
```

### 批量处理

```python
import json
from pathlib import Path

ops_dir = Path('extracted_operators')

# 遍历所有算子
for op_file in ops_dir.glob('*.json'):
    with open(op_file, 'r', encoding='utf-8') as f:
        op_data = json.load(f)
        print(f"处理算子: {op_data['kernel_name']}")
```

### 按类型筛选

```python
import json
from pathlib import Path

ops_dir = Path('extracted_operators')

# 获取所有 pointwise 算子
pointwise_ops = []
for op_file in ops_dir.glob('*.json'):
    with open(op_file, 'r', encoding='utf-8') as f:
        op_data = json.load(f)
        if op_data['func_type'] == 'pointwise':
            pointwise_ops.append(op_data)

print(f"找到 {len(pointwise_ops)} 个 pointwise 算子")
```

## 📝 算子列表

<details>
<summary>点击展开完整算子列表（130个）</summary>

- abs, add, addmm, all, amax, amin, any, arange, argmax, argmin
- attention, batch_norm, bitwise_and, bitwise_not, bitwise_or, bmm
- cat, clamp, conv1d, conv2d, conv_depthwise2d, copy, cos, count_nonzero
- cross_entropy_loss, cummin, cumsum, diag, diag_embed, diagonal, div
- dropout, elu, embedding, eq, erf, exp, exponential_, fill, flip
- full, full_like, gather, ge, gelu, groupnorm, gt, hstack, index_add
- index_put, index_select, isinf, isnan, le, log, log_softmax, logical_and
- logical_not, logical_or, lt, masked_fill, matmul, max, mean, min
- mm, mul, mv, ne, neg, normal, ones, ones_like, outer, pad
- permute, pow, prod, rand, randn, reciprocal, relu, repeat_interleave
- reshape, roll, round, rsqrt, scatter, scatter_reduce, select_scatter
- sigmoid, silu, sin, slice_scatter, softmax, sort, stack, sub, sum
- tanh, tile, topk, triu, uniform, unique, upsample_bicubic2d_aa
- upsample_nearest2d, var_mean, vdot, vector_norm, vstack, weightnorm
- where, zeros, zeros_like

</details>

## 🎓 数据来源

- **源仓库**: [FlagGems](https://github.com/FlagOpen/FlagGems) v2.2
- **提取时间**: 2025-12-18
- **提取工具**: `extract_operators.py`
- **原始文件位置**:
  - Triton 实现: `src/flag_gems/ops/`
  - 测试代码: `tests/`

## ⚙️ 提取脚本

提取脚本位于: `extract_operators.py`

### 运行方式

```bash
python extract_operators.py \
    --repo-path /path/to/FlagGems \
    --output-dir extracted_operators
```

### 主要功能

1. **算子扫描**: 自动扫描所有算子文件
2. **代码提取**: 提取完整的 Triton 实现
3. **Torch生成**: 根据签名生成对应的 Torch 实现
4. **测试转换**: 将 pytest 测试转换为 bench 格式
5. **元数据提取**: 从代码中提取参数信息

## 📖 相关资源

- [FlagGems 官方仓库](https://github.com/FlagOpen/FlagGems)
- [Triton 文档](https://triton-lang.org/)
- [PyTorch 文档](https://pytorch.org/docs/)

## 🔄 更新日志

### v1.0 (2025-12-18)
- ✅ 成功提取 130 个算子
- ✅ 修复 torch 代码变量名匹配问题
- ✅ 生成 bench 格式测试代码
- ✅ 完整的元数据提取

## 📧 联系方式

如有问题或建议，请提交 issue 或联系仓库维护者。

---

**注意**: 本数据集仅用于学习和研究目的，请遵守 FlagGems 的开源协议。
