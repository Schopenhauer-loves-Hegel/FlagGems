# 算子批量导入方案总结

**创建时间**: 2025-12-20
**分支**: feature/import-generated-ops
**状态**: ✅ 工具准备完成，等待数据

---

## 📋 方案概览

根据你的需求，设计了一个完整的批量导入流程，包含数据验证、算子筛选、批量导入三个主要阶段。

### 两批算子的筛选标准

**Batch 1: FlagGems 已有算子的优化版本**
- ✅ 标准: 你的实现相比 FlagGems 加速 **≥ 30%**
- 📊 计算方式: `speedup = flaggems_time / your_time ≥ 1.30`
- 📁 需要数据:
  - 你的算子 vs CUDA 性能
  - FlagGems 算子 vs CUDA 性能

**Batch 2: FlagGems 新增算子**
- ✅ 标准: 你的实现达到 CUDA 性能 **≥ 80%**
- 📊 计算方式: `relative = your_time / cuda_time ≤ 1.25`
- 📁 需要数据:
  - 你的算子 vs CUDA 性能

---

## 🛠️ 已创建的工具

### 1. validate_data.py - 数据验证
- **功能**: 验证性能数据格式
- **输入**: 你的性能数据 JSON + FlagGems 性能数据 JSON
- **输出**: 验证报告、数据统计、兼容性检查

### 2. filter_ops.py - 算子筛选
- **功能**: 根据性能标准筛选符合条件的算子
- **输入**: 性能数据 JSON
- **输出**: 筛选结果 JSON（包含性能指标和统计）

### 3. batch_import.py - 批量导入
- **功能**: 批量导入筛选后的算子
- **输入**: 筛选结果 JSON + 算子完整代码
- **输出**: 生成算子文件、测试文件、更新元数据

### 4. 文档
- **README.md**: 工具集概览和快速开始
- **WORKFLOW.md**: 完整工作流程（7000+ 字）
- **DATA_FORMAT.md**: 数据格式详细说明

---

## 📊 所需数据格式

### 性能数据格式

```json
{
  "operator_name": {
    "configs": [
      {
        "shape": [256, 256],
        "dtype": "float32",
        "your_time": 0.5,      // 你的实现 (ms)
        "cuda_time": 1.0       // CUDA baseline (ms)
      }
    ]
  }
}
```

### 算子完整数据格式（用于导入）

```json
{
  "op_name": "aten::gelu",
  "code": "完整的算子实现代码（Python + Triton）",
  "test_func": "完整的测试代码",
  "params": {},
  "info": {
    "total": 10,
    "success": 10,
    "failed": 0
  }
}
```

详细格式说明见：`src/flag_gems/experimental/tools/DATA_FORMAT.md`

---

## 🔄 完整工作流程

### Phase 1: 数据准备（你需要做的）

**Step 1.1: 整理性能数据**
```bash
# 创建以下 JSON 文件：
your_perf_data.json           # 你的算子性能数据
flaggems_perf_data.json       # FlagGems 算子性能数据（仅 Batch 1）
```

**Step 1.2: 准备算子完整代码**
```bash
# 为每个算子创建 JSON 文件，包含：
operator_data/
├── batch1/
│   ├── gelu.json         # 完整的实现 + 测试
│   ├── relu.json
│   └── ...
└── batch2/
    ├── huber_loss.json
    └── ...
```

**关键**: 每个 JSON 文件必须包含：
- `code`: 完整的算子实现（Python + Triton kernel）
- `test_func`: 完整的测试代码
- 其他元数据

### Phase 2: 数据验证

```bash
# 验证数据格式
python src/flag_gems/experimental/tools/validate_data.py \
    --your-data your_perf_data.json \
    --flaggems-data flaggems_perf_data.json
```

**检查点**:
- ✅ 数据格式正确
- ✅ 必需字段完整
- ✅ 数据类型正确
- ✅ 配置能够匹配（shape, dtype）

### Phase 3: 算子筛选

**Batch 1: 筛选优化版本**
```bash
python src/flag_gems/experimental/tools/filter_ops.py \
    --batch 1 \
    --your-data your_perf_data.json \
    --flaggems-data flaggems_perf_data.json \
    --output results/selected_batch1.json
```

**Batch 2: 筛选新算子**
```bash
python src/flag_gems/experimental/tools/filter_ops.py \
    --batch 2 \
    --your-data your_perf_data.json \
    --output results/selected_batch2.json
```

**检查筛选结果**:
```bash
# 查看筛选统计
cat results/selected_batch1.json | jq '{total_operators, total_configs, threshold, criterion}'

# 查看 Top 5 性能最好的算子
cat results/selected_batch1.json | jq -r '.operators | to_entries | sort_by(.value.avg_speedup_vs_flaggems) | reverse | .[0:5] | .[] | "\(.key): \(.value.avg_speedup_vs_flaggems)x"'
```

### Phase 4: 批量导入

**预览导入（不实际修改文件）**:
```bash
python src/flag_gems/experimental/tools/batch_import.py \
    --input results/selected_batch1.json \
    --batch 1 \
    --dry-run
```

**实际导入**:
```bash
python src/flag_gems/experimental/tools/batch_import.py \
    --input results/selected_batch1.json \
    --batch 1
```

### Phase 5: 测试验证

```bash
# 运行所有测试
pytest src/flag_gems/experimental/tests/ -v

# 检查元数据
python -c "
from flag_gems.experimental.metadata import MetadataManager
mgr = MetadataManager('src/flag_gems/experimental/generated/_metadata.json')
print(f'Total imported ops: {len(mgr.ops)}')
"
```

### Phase 6: 提交 PR

```bash
# 提交变更
git add src/flag_gems/experimental/
git commit -m "feat(experimental): import auto-generated operators batch 1 & 2"

# 推送
git push origin feature/import-generated-ops

# 创建 PR
gh pr create --base master --head feature/import-generated-ops
```

---

## ⚠️ 重要注意事项

### 1. 当前工具的限制

**batch_import.py 需要你适配**:
- 当前版本会为 `code` 字段生成占位符
- 你需要提供实际的算子实现代码
- 两种方案：
  1. **推荐**: 修改 `batch_import.py` 以适配你的代码存储格式
  2. 手动为每个算子创建标准格式的 JSON 文件

### 2. 数据匹配要求

- **Batch 1**: 你的数据和 FlagGems 数据中的 shape/dtype 必须完全匹配才能计算加速比
- 如果匹配的配置太少，可能需要调整测试配置

### 3. 性能数据单位

- 所有时间必须使用**毫秒 (ms)** 作为单位
- 如果你的数据是其他单位，需要预先转换

---

## 📍 下一步行动

### 你需要提供的数据

**最关键**:
1. ✅ 性能数据 JSON 文件
   - `your_perf_data.json`
   - `flaggems_perf_data.json` (for batch 1)

2. ✅ 算子完整实现
   - 每个算子的 Python + Triton 代码
   - 对应的测试代码

**建议步骤**:
1. 先找到这两个数据文件的位置
2. 检查数据格式是否符合要求（参考 DATA_FORMAT.md）
3. 运行 `validate_data.py` 验证格式
4. 告诉我数据位置，我们可以开始筛选

---

## 📚 参考文档

所有文档位于 `src/flag_gems/experimental/tools/`:

- **[README.md](src/flag_gems/experimental/tools/README.md)** - 工具集概览
- **[WORKFLOW.md](src/flag_gems/experimental/tools/WORKFLOW.md)** - 详细工作流程
- **[DATA_FORMAT.md](src/flag_gems/experimental/tools/DATA_FORMAT.md)** - 数据格式说明

---

## 🎯 成功标准

导入完成后应该满足：

- [x] 新分支 `feature/import-generated-ops` 创建
- [x] 工具和文档已提交
- [ ] 性能数据已验证
- [ ] 算子筛选完成
- [ ] 算子代码准备完整
- [ ] 批量导入成功
- [ ] 所有测试通过
- [ ] 元数据完整
- [ ] PR 创建并合并

---

## 💡 方案优势

1. **自动化**: 筛选和导入流程完全自动化
2. **可验证**: 每一步都有验证和检查
3. **可预览**: dry-run 模式避免错误
4. **可追溯**: 完整的性能数据和元数据记录
5. **可扩展**: 工具可以复用于未来的算子导入

---

**准备好数据后，我们就可以开始筛选和导入了！** 🚀
