# FlagGems Experimental Ops - TODO List

**文档版本**: v1.1
**创建日期**: 2025-12-19
**最后更新**: 2025-12-19

---

## 📋 目录

1. [分支说明](#分支说明)
2. [当前分支实现状态](#当前分支实现状态)
3. [待完成功能](#待完成功能)
4. [功能详细说明](#功能详细说明)
5. [实施计划](#实施计划)
6. [技术债务](#技术债务)

---

## 分支说明

### 当前分支: `feature/experimental-ops-framework`
**定位**: 基础设施分支，专注于完整、生产就绪的核心功能
**功能范围**:
- 元数据管理系统
- JSON 导入工具
- 直接调用接口 (`flag_gems.experimental.generated.pointwise.huber_loss()`)

### 其他分支: `feature/experimental-enable`
**定位**: Enable 功能分支，包含全局注册相关功能
**功能范围**:
- `enable()` 全局注册函数
- `disable()` 取消注册函数
- `list_enabled_ops()` 查询函数
- `is_enabled()` 状态检查函数

**注意**: Enable 功能仍需完善（unused 参数、日志、配置等），暂时保留在独立分支。

---

## 当前分支实现状态

### ✅ 已完成 (Phase 0 - 基础设施)

#### 核心基础设施
- [x] 目录结构搭建
  - [x] `src/flag_gems/experimental/` 主目录
  - [x] `generated/` 自动生成算子目录
  - [x] `custom/` 手写实验性算子目录
  - [x] `tools/` 工具目录（内置）
  - [x] `tests/` 测试目录（内置）
- [x] 元数据管理模块 (`metadata.py`)
  - [x] OpMetadata 数据结构
  - [x] OpStatus 生命周期枚举
  - [x] OpCategory 分类枚举
  - [x] MetadataManager CRUD 操作
  - [x] JSON 持久化
  - [x] 查询和过滤功能
- [x] JSON 导入工具 (`tools/import_from_json.py`)
  - [x] 从 JSON 导入算子代码
  - [x] 自动生成测试文件
  - [x] 自动注册元数据
  - [x] 自动更新 __init__.py
  - [x] 支持 category 推断和强制指定
  - [x] Dry-run 模式

#### 示例算子
- [x] huber_loss 算子成功导入
  - [x] 算子代码 (`generated/pointwise/huber_loss.py`)
  - [x] 测试代码 (`tests/test_huber_loss.py`)
  - [x] 元数据注册 (`generated/_metadata.json`)

#### 文档
- [x] RFC 文档 (`RFC.md`)
- [x] 实施文档 (`RFC_impl.md`)
- [x] TODO 文档 (本文档)

---

## 待完成功能

### ⚠️ 注意事项

本分支 (`feature/experimental-ops-framework`) 专注于基础设施的完整实现。以下列出的 enable() 相关功能已移至 `feature/experimental-enable` 分支进行开发。

---

### 🔵 Enable 功能分支待完成 (`feature/experimental-enable`)

#### 1. `enable()` 函数增强 - 添加 `unused` 参数

**分支**: `feature/experimental-enable`
**状态**: 代码已实现，需添加此功能
**优先级**: P1 (高)
**预计工作量**: 1-2 小时

**问题描述**:
目前 `enable()` 无法排除特定算子，与主分支 `flag_gems.enable(unused=[...])` 不一致。

**需求**:
```python
# 期望的 API
fg_exp.enable(
    groups=['generated'],
    unused=['huber_loss']  # 排除某些算子
)
```

**实施要点**:
- 在 `enable()` 函数签名中添加 `unused` 参数
- 传递给 `Register` 的 `user_unused_ops_list`
- 更新文档和示例

---

#### 2. `enable()` 函数增强 - 添加日志功能

**分支**: `feature/experimental-enable`
**状态**: 代码已实现，需添加此功能
**优先级**: P1 (高)
**预计工作量**: 2-3 小时

**问题描述**:
无法记录实验性算子的调用情况，调试困难。主分支有 `record`, `once`, `path` 参数。

**需求**:
```python
# 期望的 API
fg_exp.enable(
    record=True,              # 是否记录日志
    once=False,               # 是否每个位置只记录一次
    path='./exp_ops.log'      # 日志文件路径
)
```

**实施要点**:
- 复用主分支的 `setup_flaggems_logging` 函数
- 日志路径默认为 `~/.flaggems/experimental_oplist.log`
- 支持 `once` 模式（每个调用位置只记录一次）
- 在 `disable()` 时清理日志 handlers

**依赖**:
- 需要导入 `flag_gems.logging_utils.setup_flaggems_logging`

---

#### 3. `enable()` 函数增强 - 添加 `cpp_patched_ops` 参数

**分支**: `feature/experimental-enable`
**状态**: 代码已实现，需添加此功能
**优先级**: P3 (低)
**预计工作量**: 0.5 小时

**需求**:
```python
fg_exp.enable(cpp_patched_ops=['some_cpp_op'])
```

---

### 🟡 优先级 2 - 重要但不紧急 (适用于所有分支)

#### 4. 智能调度器 (Dispatcher)

**状态**: 未开始
**优先级**: P2 (中高)
**预计工作量**: 2-3 天

**问题描述**:
目前只能全局启用算子，无法根据输入特征（shape, dtype, device）智能选择实现。

**需求**:
- 根据输入特征选择最优实现
- 支持 Fallback 机制（实验算子失败时降级）
- 性能缓存（记录历史性能数据）
- 形状特化（为特定形状选择优化实现）

**实施要点**:
- 实现 `ExperimentalDispatcher` 类
- 特征提取：`_extract_features(args, kwargs)`
- 候选查找：`_find_candidates(op_name, features)`
- 最优选择：`_select_best(candidates)` 基于历史数据
- 执行与降级：`_execute_with_fallback()`
- 性能记录：`_record_performance()`

**相关设计**:
- 见 `RFC_impl.md` Module 2: 智能调度器

**依赖**:
- 需要配置管理模块 (`config.py`)

---

#### 5. 配置管理模块 (config.py)

**状态**: 未开始
**优先级**: P2 (中高)
**预计工作量**: 1 天

**问题描述**:
缺少统一的配置管理，调度策略、Fallback 行为等都是硬编码。

**需求**:
```python
from flag_gems.experimental import ExperimentalConfig

config = ExperimentalConfig(
    dispatch_strategy="safe",      # safe/aggressive/off
    fallback_on_error=True,
    fallback_on_slow=False,
    slow_threshold=1.2,
    enable_profiling=True,
    show_warnings=True,
)

fg_exp.enable(config=config)
```

**实施要点**:
- 定义 `ExperimentalConfig` dataclass
- 支持环境变量加载 (`FLAGGEMS_EXP_*`)
- 支持配置文件加载 (YAML/JSON)
- 配置验证和默认值

**相关设计**:
- 见 `RFC_impl.md` Module 4: 配置管理

---

#### 6. 异常处理模块 (exceptions.py)

**状态**: 未开始
**优先级**: P2 (中)
**预计工作量**: 0.5 天

**问题描述**:
缺少专门的异常类型，错误信息不够清晰。

**需求**:
```python
# 异常层次结构
ExperimentalError (基类)
├── MetadataError
│   ├── MetadataNotFoundError
│   ├── MetadataInvalidError
│   └── MetadataCorruptedError
├── DispatchError
│   ├── NoValidImplementationError
│   ├── FallbackFailedError
│   └── ShapeNotSupportedError
└── GraduationError
    ├── NotEligibleError
    └── ValidationFailedError
```

**相关设计**:
- 见 `RFC_impl.md` Task 1.2: 异常处理

---

### 🟢 优先级 3 - 可选功能

#### 7. 测试工具模块 (testing/)

**状态**: 未开始
**优先级**: P3 (中)
**预计工作量**: 2-3 天

**功能**:
- 精度验证工具 (`accuracy.py`)
- 性能测试工具 (`performance.py`)
- 自动化报告生成

**相关设计**:
- 见 `RFC_impl.md` Task 2.2: 实现测试工具

---

#### 8. 毕业管理系统 (graduation/)

**状态**: 未开始
**优先级**: P3 (中低)
**预计工作量**: 1 周

**功能**:
- 毕业标准检查 (`criteria.py`)
- 自动化检查工具 (`checker.py`)
- 毕业提案生成 (`proposer.py`)
- 状态追踪 (`tracker.json`)

**相关设计**:
- 见 `RFC_impl.md` Phase 3: 毕业机制

---

#### 9. CLI 管理工具

**状态**: 未开始
**优先级**: P3 (低)
**预计工作量**: 1-2 天

**功能**:
```bash
flag-gems-exp list [--filter=status]
flag-gems-exp info <op_name>
flag-gems-exp benchmark <op_name>
flag-gems-exp check-graduation <op_name>
flag-gems-exp propose-graduation <op_name>
```

**相关设计**:
- 见 `RFC_impl.md` Phase 4: 工具和 CLI

---

#### 10. `cpp_patched_ops` 参数（Enable 分支）

**分支**: `feature/experimental-enable`
**状态**: 未开始
**优先级**: P3 (低)
**预计工作量**: 0.5 小时

**问题描述**:
如果有 C++ 补丁的算子，需要排除。目前是硬编码为空列表。

**需求**:
```python
fg_exp.enable(cpp_patched_ops=['some_cpp_op'])
```

**实施要点**:
- 添加参数到 `enable()` 函数
- 传递给 `Register`
- 通常可以从配置文件读取

---

## 功能详细说明

### 当前分支完成的功能

#### 元数据管理系统

已实现完整的元数据管理功能，包括：
- OpMetadata 数据结构定义
- OpStatus 生命周期管理（EXPERIMENTAL → VALIDATED → GRADUATION_CANDIDATE → GRADUATED）
- OpCategory 分类（pointwise, reduction, blas, custom）
- MetadataManager CRUD 操作
- JSON 持久化存储

#### JSON 导入工具

已实现自动化导入工具，支持：
- 从 JSON 格式导入算子代码
- 自动推断算子类别
- 生成标准化的算子文件和测试文件
- 自动更新元数据和模块导出
- Dry-run 模式用于预览

---

### Enable 分支待实现功能

以下功能在 `feature/experimental-enable` 分支中实现：

#### Feature 1: unused 参数实现

**当前代码** (在 `feature/experimental-enable` 分支):
```python
# src/flag_gems/experimental/__init__.py
_experimental_registrar = Register(
    op_list,
    user_unused_ops_list=[],          # 🔴 硬编码
    cpp_patched_ops_list=[],
    lib=lib,
)
```

**修改后**:
```python
def enable(
    groups: Optional[List[str]] = None,
    unused: Optional[List[str]] = None,  # 🟢 新增参数
    lib: Optional[torch.library.Library] = None,
) -> None:
    # ...
    _experimental_registrar = Register(
        op_list,
        user_unused_ops_list=list(set(unused or [])),  # 🟢 使用参数
        cpp_patched_ops_list=[],
        lib=lib,
    )
```

**测试用例**:
```python
# 1. 排除单个算子
fg_exp.enable(unused=['huber_loss'])
assert 'huber_loss' not in fg_exp.list_enabled_ops()

# 2. 排除多个算子
fg_exp.enable(unused=['op1', 'op2'])
assert 'op1' not in fg_exp.list_enabled_ops()
assert 'op2' not in fg_exp.list_enabled_ops()

# 3. unused 与 groups 组合
fg_exp.enable(groups=['generated'], unused=['huber_loss'])
```

---

### Feature 2: 日志功能实现

**修改内容**:

1. **更新 `enable()` 函数签名**:
```python
def enable(
    groups: Optional[List[str]] = None,
    unused: Optional[List[str]] = None,
    lib: Optional[torch.library.Library] = None,
    record: bool = False,              # 🟢 新增
    once: bool = False,                # 🟢 新增
    path: Optional[str] = None,        # 🟢 新增
) -> None:
```

2. **在 `enable()` 末尾添加日志设置**:
```python
    # Register operators
    _experimental_registrar = Register(...)

    print(f"✅ Enabled {len(op_list)} experimental operators")

    # 🟢 Setup logging
    if record:
        from flag_gems.logging_utils import setup_flaggems_logging
        log_path = path or str(Path.home() / ".flaggems" / "experimental_oplist.log")
        setup_flaggems_logging(path=log_path, record=record, once=once)
        print(f"📝 Logging to: {log_path}")
```

3. **更新 `disable()` 函数清理日志**:
```python
def disable() -> None:
    global _experimental_lib, _experimental_registrar

    # ... existing cleanup code ...

    # 🟢 Clean up logging handlers
    import logging
    logger = logging.getLogger("flag_gems")
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    print("✅ Experimental operators disabled")
```

**测试用例**:
```python
import tempfile
from pathlib import Path

# 1. 测试基础日志
with tempfile.NamedTemporaryFile() as f:
    fg_exp.enable(record=True, path=f.name)
    # ... 使用算子 ...
    fg_exp.disable()

    # 检查日志文件
    log_content = Path(f.name).read_text()
    assert "huber_loss" in log_content

# 2. 测试 once 模式
fg_exp.enable(record=True, once=True)
# 多次调用同一算子，只应记录一次

# 3. 测试默认路径
fg_exp.enable(record=True)
assert Path.home() / ".flaggems" / "experimental_oplist.log" exists
```

---

## 实施计划

### 当前分支 (`feature/experimental-ops-framework`)

#### ✅ Phase 0: 基础设施 - 已完成

**状态**: 已完成并可以合并到主分支

**已完成内容**:
1. ✅ 目录结构搭建
2. ✅ 元数据管理模块 (`metadata.py`)
3. ✅ JSON 导入工具 (`tools/import_from_json.py`)
4. ✅ 示例算子导入 (`huber_loss`)
5. ✅ 文档和 RFC

**验收标准** (全部满足):
- ✅ 可以通过 JSON 导入算子
- ✅ 可以通过 `flag_gems.experimental.generated.pointwise.huber_loss()` 直接调用算子
- ✅ 元数据正确注册和查询
- ✅ 代码结构清晰，文档完整

---

### Enable 功能分支 (`feature/experimental-enable`)

#### 🔵 Phase 1.1: enable() 函数完善

**状态**: 基础实现已完成，待增强功能
**分支**: `feature/experimental-enable`

**时间**: 1-2 天
**任务**:
1. 添加 `unused` 参数 (2 小时)
2. 添加日志功能 (3 小时)
3. 添加 `cpp_patched_ops` 参数 (1 小时)
4. 更新文档和示例 (2 小时)
5. 编写单元测试 (2 小时)

**验收标准**:
- [ ] `unused` 参数工作正常
- [ ] 日志功能正常记录
- [ ] API 与主分支一致（除了新增的 groups）
- [ ] 测试覆盖率 > 80%
- [ ] 文档更新完整

---

### 未来开发计划 (跨分支)

#### Phase 1.2: 基础设施扩展

**适用分支**: 新建特性分支
**时间**: 2-3 天
**任务**:
1. 实现配置管理模块 (1 天)
2. 实现异常处理模块 (0.5 天)
3. 完善错误处理和边界情况 (0.5 天)
4. 文档和测试 (1 天)

---

#### Phase 2: 核心功能 - 智能调度器

**适用分支**: 新建特性分支

**时间**: 3-5 天
**任务**:
1. 实现 ExperimentalDispatcher (2-3 天)
2. 实现性能缓存 (1 天)
3. 集成到 enable() (0.5 天)
4. 测试和文档 (1 天)

---

### Phase 3+: 其他功能

根据实际需求和优先级决定。

---

## 技术债务

### Debt 1: 硬编码的算子列表

**位置**: `_build_op_registration_list()`
**问题**: 虽然是动态加载，但过滤逻辑是硬编码的
**影响**: 中
**建议**: 将过滤规则配置化

---

### Debt 2: 错误处理不完善

**位置**: 多处
**问题**: 使用 print 输出错误，没有专门的异常类型
**影响**: 中
**建议**: 实现 exceptions.py 模块

---

### Debt 3: 缺少单元测试

**位置**: `tests/experimental/`
**问题**: 只有手动测试，没有自动化测试
**影响**: 高
**建议**: 添加完整的测试套件

---

### Debt 4: 文档不完整

**位置**: `docs/experimental/`
**问题**: 只有示例代码，缺少详细的用户文档
**影响**: 中
**建议**: 编写完整的用户指南和 API 文档

---

## 附录

### A. 相关文件清单

**核心文件**:
- `src/flag_gems/experimental/__init__.py` - 主模块
- `src/flag_gems/experimental/metadata.py` - 元数据管理
- `src/flag_gems/experimental/tools/import_from_json.py` - 导入工具

**待创建文件**:
- `src/flag_gems/experimental/config.py` - 配置管理
- `src/flag_gems/experimental/exceptions.py` - 异常处理
- `src/flag_gems/experimental/dispatcher.py` - 智能调度器

**文档文件**:
- `RFC.md` - RFC 机制设计
- `RFC_impl.md` - 实施文档
- `TODO.md` - 本文档

---

### B. 参考资料

- 主分支 enable() 实现: `src/flag_gems/__init__.py:29-362`
- Register 类实现: `src/flag_gems/runtime/register.py`
- 日志工具: `src/flag_gems/logging_utils.py`
- RFC 设计文档: `RFC.md` 和 `RFC_impl.md`

---

### C. 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2025-12-19 | v1.0 | 初始版本，列出所有待完成功能 |
| 2025-12-19 | v1.1 | 拆分分支说明：当前分支专注基础设施，enable() 功能移至独立分支 |

---

**文档结束**

_此文档将持续更新，记录功能实现进度和新增需求。_
