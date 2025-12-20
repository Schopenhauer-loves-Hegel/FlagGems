# 算子筛选工具更新说明

## ✅ 已完成的修改

### filter_ops.py - 完全重写

#### 主要变更

1. **数据源适配**
   - ✅ 支持 GPT 数据目录格式（speedup_summary.json + log_X/result.json）
   - ✅ 支持 FlagGems Excel 文件（vendor-test-1106.xlsx, Speedup sheet）
   - ✅ 自动从 result.json 提取算子代码

2. **计算公式修正**
   - ✅ 正确的相对加速比：`gpt_speedup / flaggems_speedup`
   - ✅ 当结果 >= threshold 时表示 GPT 比 FlagGems 快

3. **阈值可配置**
   - ✅ 通过 `--threshold` 参数自定义阈值
   - ✅ 默认值：Batch 1 = 1.30，Batch 2 = 0.80

4. **改进的输出**
   - ✅ 详细的日志信息
   - ✅ 筛选摘要和 Top 10 列表
   - ✅ JSON 格式的完整结果

---

## 📝 新的使用方式

### Batch 1: 筛选优于 FlagGems 的算子

```bash
cd /share/project/tj/fork/FlagGems

python src/flag_gems/experimental/tools/filter_ops.py \
    --batch 1 \
    --gpt-data-dir src/flag_gems/experimental/data/eval_perf_gpt5_pass_10_20251117-114806 \
    --flaggems-excel src/flag_gems/experimental/data/vendor-test-1106.xlsx \
    --threshold 1.2 \
    --output src/flag_gems/experimental/data/results/selected_batch1.json
```

**参数说明**：
- `--batch 1`: Batch 1 模式（比较 GPT vs FlagGems）
- `--gpt-data-dir`: GPT 性能数据目录
- `--flaggems-excel`: FlagGems Excel 文件
- `--threshold 1.2`: 自定义阈值（GPT 比 FlagGems 快 >= 20%）
- `--output`: 输出文件路径

### Batch 2: 筛选达到 CUDA 性能的新算子

```bash
python src/flag_gems/experimental/tools/filter_ops.py \
    --batch 2 \
    --gpt-data-dir <your_gpt_data_dir> \
    --threshold 0.8 \
    --output selected_batch2.json
```

---

## 📊 输出格式

筛选结果 JSON：

```json
{
  "batch": 1,
  "threshold": 1.2,
  "criterion": "speedup_vs_flaggems",
  "total_operators": 63,
  "selected_operators": 3,
  "operators": {
    "sort": {
      "gpt_speedup_vs_cuda": 1.0368,
      "flaggems_speedup_vs_cuda": 0.0629,
      "speedup_vs_flaggems": 16.4935,
      "code": "完整的 Triton 代码...",
      "has_code": true
    },
    "randperm": {
      "gpt_speedup_vs_cuda": 1.1673,
      "flaggems_speedup_vs_cuda": 0.5765,
      "speedup_vs_flaggems": 2.0247,
      "code": "完整的 Triton 代码...",
      "has_code": true
    }
    // ... 更多算子
  }
}
```

---

## 🎯 阈值建议

### Batch 1 阈值选择

| 阈值 | 含义 | 预期结果 |
|------|------|----------|
| 1.5 | GPT 快 50% | 极少数优秀算子 |
| 1.3 | GPT 快 30% | 少量算子（高质量） |
| 1.2 | GPT 快 20% | 中等数量 |
| 1.1 | GPT 快 10% | 较多算子 |
| 1.0 | GPT 稍快 | 大量算子（可能提升不明显） |

### Batch 2 阈值选择

| 阈值 | 含义 | 说明 |
|------|------|------|
| 1.0 | 100% CUDA 性能 | 与 CUDA 相当或更快 |
| 0.9 | 90% CUDA 性能 | 略慢于 CUDA |
| 0.8 | 80% CUDA 性能 | 默认阈值 |
| 0.7 | 70% CUDA 性能 | 较宽松 |

---

## 🔄 等待新数据

当前数据有问题，等待你提供新的准确数据后：

1. 将新数据放到 `src/flag_gems/experimental/data/` 目录
2. 运行上述筛选命令
3. 查看筛选结果
4. 根据需要调整阈值
5. 继续后续的批量导入流程

---

## 📁 数据目录结构

```
src/flag_gems/experimental/data/
├── <your_new_gpt_data>/           # 你的新 GPT 数据目录
│   ├── speedup_summary.json       # 汇总文件
│   └── log_X/                     # 详细数据
│       └── result.json            # 包含代码
│
├── vendor-test-1106.xlsx          # FlagGems 性能数据
│
└── results/                       # 筛选结果
    └── selected_batch1.json       # 输出文件
```

---

## ✅ 下一步

数据准备好后的工作流程：

1. **筛选算子** ✅（脚本已完成）
   ```bash
   python filter_ops.py --batch 1 \
       --gpt-data-dir <new_data> \
       --flaggems-excel vendor-test-1106.xlsx \
       --threshold 1.2 \
       --output selected_batch1.json
   ```

2. **查看结果**
   ```bash
   cat selected_batch1.json | jq '.selected_operators'
   cat selected_batch1.json | jq '.operators | keys'
   ```

3. **批量导入**（需要修改 batch_import.py）
   - 适配新的输入格式
   - 跳过测试生成（使用 FlagGems 测试）
   - 生成算子文件
   - 更新元数据

---

## 🐛 问题排查

### 检查数据格式

```bash
# 检查 GPT 数据
ls <gpt_data_dir>/
cat <gpt_data_dir>/speedup_summary.json | jq '.statistics'

# 检查 Excel 文件
python3 -c "
import pandas as pd
df = pd.read_excel('vendor-test-1106.xlsx', sheet_name='Speedup')
print(df.head())
"
```

### 测试筛选脚本

```bash
# 用当前数据测试（即使不准确）
python src/flag_gems/experimental/tools/filter_ops.py \
    --batch 1 \
    --gpt-data-dir src/flag_gems/experimental/data/eval_perf_gpt5_pass_10_20251117-114806 \
    --flaggems-excel src/flag_gems/experimental/data/vendor-test-1106.xlsx \
    --threshold 1.2 \
    --output /tmp/test_output.json

# 查看结果
cat /tmp/test_output.json | jq '.'
```

---

**准备好新数据后告诉我，我们继续！** 🚀
