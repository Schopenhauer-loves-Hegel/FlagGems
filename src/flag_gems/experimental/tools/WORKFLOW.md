# 算子批量导入工作流程

## 📋 目录

1. [前置准备](#前置准备)
2. [Batch 1: FlagGems 已有算子优化版本](#batch-1-flaggems-已有算子优化版本)
3. [Batch 2: FlagGems 新增算子](#batch-2-flaggems-新增算子)
4. [验证和测试](#验证和测试)
5. [提交 PR](#提交-pr)

---

## 前置准备

### 1. 确认环境

```bash
# 确认在正确的分支
git branch
# 应该显示: * feature/import-generated-ops

# 确认工作区干净
git status
```

### 2. 数据准备

参考 `DATA_FORMAT.md` 准备以下数据：

**必需文件**：
- `your_perf_data.json` - 你的算子性能数据
- `flaggems_perf_data.json` - FlagGems 算子性能数据（仅 Batch 1）
- `operator_data/` - 每个算子的完整实现和测试代码

**目录结构示例**：
```
/path/to/data/
├── your_perf_data.json          # 你的算子性能
├── flaggems_perf_data.json      # FlagGems 性能
└── operator_data/
    ├── batch1/
    │   ├── gelu.json            # 每个算子的完整代码
    │   ├── relu.json
    │   └── ...
    └── batch2/
        ├── huber_loss.json
        └── ...
```

---

## Batch 1: FlagGems 已有算子优化版本

### 目标
筛选出比 FlagGems 现有实现快 **≥30%** 的算子

### Step 1: 筛选符合条件的算子

```bash
cd /share/project/tj/fork/FlagGems

# 运行筛选脚本
python src/flag_gems/experimental/tools/filter_ops.py \
    --batch 1 \
    --your-data /path/to/your_perf_data.json \
    --flaggems-data /path/to/flaggems_perf_data.json \
    --output results/selected_batch1.json
```

**输出**：
- `results/selected_batch1.json` - 符合条件的算子列表及性能数据

**检查结果**：
```bash
# 查看筛选结果
cat results/selected_batch1.json | jq '.total_operators'

# 查看 Top 5 性能提升最大的算子
cat results/selected_batch1.json | jq -r '.operators | to_entries | sort_by(.value.avg_speedup_vs_flaggems) | reverse | .[0:5] | .[] | "\(.key): \(.value.avg_speedup_vs_flaggems)x"'
```

### Step 2: 准备算子完整数据

为筛选出的每个算子准备 JSON 文件（包含 code 和 test_func）：

```bash
# 假设你的算子代码在某个目录
# 需要将它们转换为 JSON 格式

# 示例：为 gelu 创建 JSON
python -c "
import json
with open('operator_data/batch1/gelu.json', 'w') as f:
    json.dump({
        'op_name': 'aten::gelu',
        'code': open('path/to/gelu_implementation.py').read(),
        'test_func': open('path/to/gelu_test.py').read(),
        'params': {},
        'info': {'total': 10, 'success': 10, 'failed': 0}
    }, f, indent=2)
"
```

### Step 3: 批量导入（先预览）

```bash
# 预览导入（不实际修改文件）
python src/flag_gems/experimental/tools/batch_import.py \
    --input results/selected_batch1.json \
    --batch 1 \
    --dry-run
```

**检查预览输出**，确认：
- ✅ 所有算子都能正确识别分类（pointwise/reduction/blas）
- ✅ 没有与现有算子冲突
- ✅ 文件路径正确

### Step 4: 实际导入

```bash
# 实际导入
python src/flag_gems/experimental/tools/batch_import.py \
    --input results/selected_batch1.json \
    --batch 1
```

**注意**：当前版本的 `batch_import.py` 需要你提供每个算子的完整实现。
如果你的算子代码不在标准 JSON 格式中，需要先转换。

---

## Batch 2: FlagGems 新增算子

### 目标
筛选出达到 CUDA 性能 **≥80%** 的新算子

### Step 1: 筛选符合条件的算子

```bash
# 运行筛选脚本
python src/flag_gems/experimental/tools/filter_ops.py \
    --batch 2 \
    --your-data /path/to/your_perf_data.json \
    --output results/selected_batch2.json
```

**输出**：
- `results/selected_batch2.json` - 符合条件的算子列表

**检查结果**：
```bash
# 查看筛选结果
cat results/selected_batch2.json | jq '.total_operators'

# 查看最接近 CUDA 性能的 Top 5
cat results/selected_batch2.json | jq -r '.operators | to_entries | sort_by(.value.avg_relative_to_cuda) | .[0:5] | .[] | "\(.key): \(.value.avg_relative_to_cuda | (1 / . * 100))% of CUDA"'
```

### Step 2-4: 与 Batch 1 相同

准备数据 → 预览导入 → 实际导入

```bash
# 预览
python src/flag_gems/experimental/tools/batch_import.py \
    --input results/selected_batch2.json \
    --batch 2 \
    --dry-run

# 实际导入
python src/flag_gems/experimental/tools/batch_import.py \
    --input results/selected_batch2.json \
    --batch 2
```

---

## 验证和测试

### 1. 检查导入结果

```bash
# 查看导入的算子文件
find src/flag_gems/experimental/generated -name "*.py" -type f | grep -v __pycache__ | grep -v __init__

# 查看元数据
cat src/flag_gems/experimental/generated/_metadata.json | jq '.ops | length'
cat src/flag_gems/experimental/generated/_metadata.json | jq '.ops | keys'
```

### 2. 运行测试

```bash
# 运行所有 experimental 测试
pytest src/flag_gems/experimental/tests/ -v

# 运行特定算子测试
pytest src/flag_gems/experimental/tests/test_gelu.py -v

# 运行性能测试（如果有）
pytest src/flag_gems/experimental/tests/ -v -m benchmark
```

### 3. 验证导入的算子

```python
# 测试导入和调用
python -c "
import torch
from flag_gems.experimental.generated.pointwise import gelu

x = torch.randn(256, 256, device='cuda')
result = gelu(x)
print(f'✓ gelu works! Output shape: {result.shape}')
"
```

### 4. 检查元数据完整性

```bash
# 验证元数据
python -c "
from flag_gems.experimental.metadata import MetadataManager

mgr = MetadataManager('src/flag_gems/experimental/generated/_metadata.json')
print(f'Total ops: {len(mgr.ops)}')

# 检查每个算子的元数据
for op_id, metadata in mgr.ops.items():
    print(f\"  - {metadata['op_name']}: {metadata['category']} ({metadata['status']})\")
"
```

---

## 提交 PR

### 1. 查看变更

```bash
git status
git diff --stat

# 查看具体变更
git diff src/flag_gems/experimental/
```

### 2. 提交变更

```bash
# 添加所有变更
git add src/flag_gems/experimental/

# 提交（根据批次选择消息）
# Batch 1
git commit -m "feat(experimental): import batch1 ops with 30%+ speedup vs FlagGems

- Imported X operators from batch 1
- All ops show ≥30% speedup over existing FlagGems implementations
- Added tests and metadata for all ops
"

# Batch 2
git commit -m "feat(experimental): import batch2 new ops with 80%+ CUDA performance

- Imported Y operators from batch 2
- All ops achieve ≥80% of CUDA baseline performance
- Added tests and metadata for all ops
"

# 或者两批一起
git commit -m "feat(experimental): import auto-generated operators (batch 1 & 2)

## Batch 1: Optimized existing ops (X operators)
- Criterion: ≥30% speedup vs FlagGems
- Categories: pointwise (A), reduction (B), blas (C)

## Batch 2: New operators (Y operators)
- Criterion: ≥80% of CUDA performance
- Categories: pointwise (D), reduction (E), blas (F)

## Summary
- Total imported: X+Y operators
- All tests passing
- Metadata complete
"
```

### 3. 推送到远程

```bash
# 推送分支
git push origin feature/import-generated-ops
```

### 4. 创建 PR

```bash
# 使用 gh CLI 创建 PR
gh pr create \
    --base master \
    --head feature/import-generated-ops \
    --title "feat(experimental): 批量导入自动生成算子" \
    --body "$(cat <<'EOF'
## 📦 概述

本 PR 批量导入自动生成的算子到 experimental 框架。

## 🎯 导入标准

### Batch 1: FlagGems 已有算子优化版本
- **标准**: 相比 FlagGems 现有实现加速 ≥30%
- **数量**: X 个算子
- **分类**:
  - Pointwise: A 个
  - Reduction: B 个
  - BLAS: C 个

### Batch 2: FlagGems 新增算子
- **标准**: 达到 CUDA 性能的 ≥80%
- **数量**: Y 个算子
- **分类**:
  - Pointwise: D 个
  - Reduction: E 个
  - BLAS: F 个

## 📊 性能数据

详细性能数据见筛选结果：
- `results/selected_batch1.json`
- `results/selected_batch2.json`

## ✅ 检查清单

- [x] 所有算子已通过筛选标准
- [x] 代码符合 FlagGems 规范
- [x] 包含完整测试
- [x] 元数据注册完整
- [x] 测试通过

## 🧪 测试

运行测试：
\`\`\`bash
pytest src/flag_gems/experimental/tests/ -v
\`\`\`

## 📝 后续工作

- [ ] 添加更多测试配置（不同 shape/dtype）
- [ ] 性能benchmark
- [ ] 文档更新
EOF
)"
```

---

## 故障排查

### 问题 1: 筛选脚本找不到匹配的配置

**症状**：筛选结果为空或很少

**原因**：你的数据和 FlagGems 数据中的 shape/dtype 不匹配

**解决**：
```bash
# 检查数据格式
python -c "
import json
your_data = json.load(open('your_perf_data.json'))
fg_data = json.load(open('flaggems_perf_data.json'))

# 查看某个算子的配置
print('Your configs:', your_data['gelu']['configs'][0])
print('FlagGems configs:', fg_data['gelu']['configs'][0])
"
```

### 问题 2: 导入失败 - 缺少代码

**症状**：`batch_import.py` 报错 "TODO: Add actual implementation"

**原因**：没有提供算子的完整实现代码

**解决**：为每个算子创建包含 `code` 和 `test_func` 的 JSON 文件

### 问题 3: 测试失败

**症状**：导入的算子测试不通过

**解决**：
```bash
# 单独测试问题算子
pytest src/flag_gems/experimental/tests/test_<op_name>.py -v -s

# 查看详细错误
python -c "
import torch
from flag_gems.experimental.generated.<category>.<op_name> import <op_name>

x = torch.randn(10, device='cuda')
try:
    result = <op_name>(x)
    print('Success!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
"
```

---

## 性能追踪

建议记录导入过程的关键指标：

```bash
# 创建导入日志
echo "Import started: $(date)" >> import_log.txt
echo "Batch 1: X ops" >> import_log.txt
echo "Batch 2: Y ops" >> import_log.txt

# 记录测试结果
pytest src/flag_gems/experimental/tests/ -v > test_results.txt 2>&1

# 记录性能数据
# TODO: 运行 benchmark 并保存结果
```

---

## 参考

- `DATA_FORMAT.md` - 数据格式说明
- `filter_ops.py` - 筛选脚本
- `batch_import.py` - 批量导入脚本
- `import_from_json.py` - 单个算子导入
- `TODO.md` - 项目待办事项
