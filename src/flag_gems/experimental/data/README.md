# Experimental Data Directory

此目录用于存放算子导入相关的原始数据和中间结果。

## 📁 目录结构

```
data/
├── batch1/                      # Batch 1 原始数据（FlagGems 已有算子优化版本）
│   ├── your_perf_data.json     # 你的算子性能数据
│   ├── flaggems_perf_data.json # FlagGems 算子性能数据
│   └── operator_code/          # 算子完整实现代码（JSON 格式）
│       ├── gelu.json
│       ├── relu.json
│       └── ...
│
├── batch2/                      # Batch 2 原始数据（FlagGems 新增算子）
│   ├── your_perf_data.json     # 你的算子性能数据
│   └── operator_code/          # 算子完整实现代码（JSON 格式）
│       ├── huber_loss.json
│       └── ...
│
└── results/                     # 筛选和处理结果
    ├── selected_batch1.json    # Batch 1 筛选结果
    ├── selected_batch2.json    # Batch 2 筛选结果
    ├── validation_report.txt   # 数据验证报告
    └── import_summary.txt      # 导入总结报告
```

## 📊 数据文件说明

### 性能数据格式 (your_perf_data.json / flaggems_perf_data.json)

```json
{
  "operator_name": {
    "configs": [
      {
        "shape": [256, 256],
        "dtype": "float32",
        "your_time": 0.5,        // 你的实现耗时 (ms)
        "cuda_time": 1.0,        // CUDA baseline (ms)
        "flaggems_time": 0.7     // FlagGems 实现 (仅 batch1 需要)
      }
    ]
  }
}
```

### 算子完整代码格式 (operator_code/*.json)

```json
{
  "op_name": "aten::gelu",
  "code": "完整的 Python + Triton 实现代码",
  "test_func": "完整的测试代码",
  "params": {},
  "info": {
    "total": 10,
    "success": 10,
    "failed": 0
  }
}
```

详细格式说明请参考: `../tools/DATA_FORMAT.md`

## 🔄 使用流程

### 1. 放置原始数据

将你的数据文件放到对应目录：

```bash
# Batch 1 数据
cp /path/to/your_perf.json data/batch1/your_perf_data.json
cp /path/to/flaggems_perf.json data/batch1/flaggems_perf_data.json

# Batch 2 数据
cp /path/to/your_perf.json data/batch2/your_perf_data.json
```

### 2. 放置算子代码

```bash
# 将算子实现放到对应的 operator_code 目录
mkdir -p data/batch1/operator_code
cp /path/to/operators/*.json data/batch1/operator_code/
```

### 3. 验证数据

```bash
# 验证 Batch 1
python ../tools/validate_data.py \
    --your-data data/batch1/your_perf_data.json \
    --flaggems-data data/batch1/flaggems_perf_data.json

# 验证 Batch 2
python ../tools/validate_data.py \
    --your-data data/batch2/your_perf_data.json
```

### 4. 筛选算子

```bash
# 筛选 Batch 1
python ../tools/filter_ops.py \
    --batch 1 \
    --your-data data/batch1/your_perf_data.json \
    --flaggems-data data/batch1/flaggems_perf_data.json \
    --output data/results/selected_batch1.json

# 筛选 Batch 2
python ../tools/filter_ops.py \
    --batch 2 \
    --your-data data/batch2/your_perf_data.json \
    --output data/results/selected_batch2.json
```

### 5. 查看筛选结果

```bash
# 查看统计
cat data/results/selected_batch1.json | jq '{total_operators, total_configs}'

# 查看具体算子
cat data/results/selected_batch1.json | jq '.operators | keys'
```

## 📝 注意事项

### Git 管理

- **原始数据文件通常很大，不建议提交到 git**
- 已配置 `.gitignore` 忽略数据文件
- 仅保留示例文件和文档
- 筛选结果（JSON）可以选择性提交

### 数据备份

建议在本地保留数据备份：

```bash
# 备份原始数据
tar -czf experimental_data_backup_$(date +%Y%m%d).tar.gz data/
```

### 数据清理

筛选和导入完成后，可以清理中间文件：

```bash
# 清理结果文件（保留原始数据）
rm -f data/results/*

# 完全清理（谨慎！）
# rm -rf data/batch1/* data/batch2/* data/results/*
```

## 🔗 相关文档

- **工具使用**: `../tools/README.md`
- **完整流程**: `../tools/WORKFLOW.md`
- **数据格式**: `../tools/DATA_FORMAT.md`

---

**准备好数据后，就可以开始筛选和导入流程了！** 🚀
