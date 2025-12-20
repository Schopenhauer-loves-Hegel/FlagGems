# 算子导入数据格式说明

## 概述

本文档说明算子筛选和导入所需的数据格式。

---

## 1. 性能数据格式（输入）

### 1.1 你的算子性能数据格式

**文件**: `your_perf_data.json`

```json
{
  "operator_name_1": {
    "configs": [
      {
        "shape": [256, 256],
        "dtype": "float32",
        "your_time": 0.5,      // 你的实现耗时 (ms)
        "cuda_time": 1.0       // CUDA baseline 耗时 (ms)
      },
      {
        "shape": [512, 512],
        "dtype": "float16",
        "your_time": 0.8,
        "cuda_time": 1.5
      }
    ]
  },
  "operator_name_2": {
    "configs": [...]
  }
}
```

### 1.2 FlagGems 算子性能数据格式（仅 Batch 1 需要）

**文件**: `flaggems_perf_data.json`

```json
{
  "operator_name_1": {
    "configs": [
      {
        "shape": [256, 256],
        "dtype": "float32",
        "flaggems_time": 0.7,  // FlagGems 现有实现耗时 (ms)
        "cuda_time": 1.0       // CUDA baseline 耗时 (ms)
      },
      {
        "shape": [512, 512],
        "dtype": "float16",
        "flaggems_time": 1.0,
        "cuda_time": 1.5
      }
    ]
  }
}
```

**重要说明**：
- `shape` 和 `dtype` 必须在两个文件中完全匹配才能计算相对加速比
- `cuda_time` 应该是同一个 baseline（理想情况下是相同的测试环境）

---

## 2. 筛选结果格式（中间输出）

**文件**: `selected_batch1.json` 或 `selected_batch2.json`

```json
{
  "batch": 1,
  "threshold": 1.30,
  "criterion": "speedup_vs_flaggems",
  "total_operators": 15,
  "total_configs": 45,
  "operators": {
    "gelu": {
      "avg_speedup_vs_flaggems": 1.45,      // 平均加速比（vs FlagGems）
      "avg_relative_to_cuda": 0.85,         // 平均相对 CUDA 性能
      "configs": [
        {
          "shape": [256, 256],
          "dtype": "float32",
          "your_time": 0.5,
          "cuda_time": 1.0,
          "flaggems_time": 0.7,
          "speedup_vs_flaggems": 1.4,       // 0.7 / 0.5 = 1.4x
          "relative_to_cuda": 0.5           // 0.5 / 1.0 = 0.5 (50%)
        }
      ]
    }
  }
}
```

---

## 3. 算子完整数据格式（用于导入）

### 3.1 基本结构

为了实际导入算子，你需要提供每个算子的**完整实现代码**和**测试代码**。

**文件**: `operator_name.json` (每个算子一个文件)

```json
{
  "op_name": "aten::gelu",
  "code": "完整的算子实现代码...",
  "test_func": "完整的测试代码...",
  "params": {
    "approximate": "none"
  },
  "info": {
    "total": 10,
    "success": 10,
    "failed": 0
  }
}
```

### 3.2 代码字段说明

#### `code` 字段 - 算子实现

应该包含完整的 Python + Triton 实现：

```python
import torch
import triton
import triton.language as tl
from flag_gems.utils import libentry

@libentry()
@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Kernel implementation
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    # GELU calculation
    result = 0.5 * x * (1.0 + tl.math.tanh(0.797885 * (x + 0.044715 * x * x * x)))
    tl.store(output_ptr + offsets, result, mask=mask)

def gelu(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = input.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)

    return output
```

#### `test_func` 字段 - 测试代码

应该包含完整的测试函数：

```python
import torch
import pytest
from flag_gems.experimental.generated.pointwise.gelu import gelu

def test_gelu_accuracy():
    shapes = [(256,), (256, 256), (8, 128, 128)]
    dtypes = [torch.float32, torch.float16]

    for shape in shapes:
        for dtype in dtypes:
            x = torch.randn(shape, dtype=dtype, device='cuda')

            # Your implementation
            result = gelu(x)

            # Reference (PyTorch)
            expected = torch.nn.functional.gelu(x)

            # Check accuracy
            torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)

def test_gelu_shapes():
    # Test various shapes
    shapes = [(1,), (1000,), (256, 256), (8, 128, 128, 128)]

    for shape in shapes:
        x = torch.randn(shape, device='cuda')
        result = gelu(x)
        assert result.shape == shape
```

---

## 4. 数据准备工作流程

### Step 1: 整理性能数据

将你的测试结果整理成上述 JSON 格式：

```bash
# 你需要创建这两个文件
your_perf_data.json           # 你的算子性能
flaggems_perf_data.json       # FlagGems 算子性能（batch 1）
```

### Step 2: 整理算子代码

为每个算子创建完整的 JSON 文件，包含：
- 算子实现代码 (`code` 字段)
- 测试代码 (`test_func` 字段)

可以组织成目录结构：

```
operator_data/
├── batch1/
│   ├── gelu.json
│   ├── relu.json
│   └── ...
└── batch2/
    ├── huber_loss.json
    ├── smooth_l1_loss.json
    └── ...
```

---

## 5. 示例：完整的 gelu.json

```json
{
  "op_name": "aten::gelu",
  "code": "import torch\nimport triton\nimport triton.language as tl\nfrom flag_gems.utils import libentry\n\n@libentry()\n@triton.jit\ndef gelu_kernel(\n    input_ptr,\n    output_ptr,\n    n_elements,\n    BLOCK_SIZE: tl.constexpr,\n):\n    pid = tl.program_id(0)\n    block_start = pid * BLOCK_SIZE\n    offsets = block_start + tl.arange(0, BLOCK_SIZE)\n    mask = offsets < n_elements\n    \n    x = tl.load(input_ptr + offsets, mask=mask)\n    result = 0.5 * x * (1.0 + tl.math.tanh(0.797885 * (x + 0.044715 * x * x * x)))\n    tl.store(output_ptr + offsets, result, mask=mask)\n\ndef gelu(input: torch.Tensor, approximate: str = \"none\") -> torch.Tensor:\n    output = torch.empty_like(input)\n    n_elements = input.numel()\n    \n    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)\n    gelu_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024)\n    \n    return output\n",
  "test_func": "import torch\nimport pytest\nfrom flag_gems.experimental.generated.pointwise.gelu import gelu\n\ndef test_gelu_accuracy():\n    shapes = [(256,), (256, 256), (8, 128, 128)]\n    dtypes = [torch.float32, torch.float16]\n    \n    for shape in shapes:\n        for dtype in dtypes:\n            x = torch.randn(shape, dtype=dtype, device='cuda')\n            result = gelu(x)\n            expected = torch.nn.functional.gelu(x)\n            torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)\n",
  "params": {
    "approximate": "none"
  },
  "info": {
    "total": 6,
    "success": 6,
    "failed": 0
  }
}
```

---

## 6. 数据准备检查清单

在运行筛选脚本之前，确保：

- [ ] **Batch 1（FlagGems 已有算子）**
  - [ ] `your_perf_data.json` - 包含你的算子性能数据
  - [ ] `flaggems_perf_data.json` - 包含 FlagGems 算子性能数据
  - [ ] 两个文件中的测试配置 (shape, dtype) 能够匹配
  - [ ] 每个算子有对应的完整 JSON 文件（含 code 和 test_func）

- [ ] **Batch 2（FlagGems 没有的算子）**
  - [ ] `your_perf_data.json` - 包含你的算子性能数据
  - [ ] 每个算子有对应的完整 JSON 文件（含 code 和 test_func）

---

## 7. 常见问题

### Q1: 性能数据的单位是什么？

A: 时间单位统一使用**毫秒 (ms)**。如果你的数据是其他单位，需要转换。

### Q2: 如果同一个算子有多个实现版本怎么办？

A: 选择性能最好的版本，或者使用不同的命名（如 `gelu_v1`, `gelu_v2`）。

### Q3: 测试代码必须包含吗？

A: 强烈建议包含。如果暂时没有，可以先用占位符，后续补充。

### Q4: 如何处理不同 shape/dtype 的性能差异？

A: 筛选脚本会计算平均性能。如果某个配置特别突出，可以在注释中说明。

---

## 8. 下一步

数据准备完成后，参考 `WORKFLOW.md` 运行筛选和导入流程。
