# FlagGems Experimental Ops 实施文档

## 文档信息
- **文档版本**: v1.0
- **创建日期**: 2025-12-19
- **状态**: DRAFT
- **相关 RFC**: RFC.md - FlagGems 自动生成算子准入机制

## 目录
1. [项目概述](#项目概述)
2. [架构设计](#架构设计)
3. [模块详细设计](#模块详细设计)
4. [开发任务清单](#开发任务清单)
5. [实施计划](#实施计划)
6. [技术规范](#技术规范)
7. [测试策略](#测试策略)
8. [文档和示例](#文档和示例)

---

## 项目概述

### 目标
为 FlagGems 实现一个完整的 experimental ops 机制，支持：
- 自动生成算子的规范化管理
- 实验性算子与稳定算子的隔离
- 智能调度和性能优化
- 标准化的毕业流程

### 核心价值
1. **快速迭代**：允许未成熟算子快速上线，接受社区验证
2. **质量保障**：通过隔离机制保护主分支稳定性
3. **透明度**：清晰的状态追踪和毕业标准
4. **自动化**：减少人工审核负担，提升开发效率

### 设计原则
- **显式优于隐式**：用户必须明确选择使用实验性功能
- **安全降级**：任何失败都能回退到稳定实现
- **可观测性**：完整的日志和性能追踪
- **自动化优先**：最大化自动化验证和决策

---

## 架构设计

### 1. 目录结构

```
src/flag_gems/
├── __init__.py                          # 主入口（修改）
├── ops/                                 # 稳定算子（已存在）
│   ├── __init__.py
│   └── *.py
├── experimental/                        # 新增：实验性功能根目录
│   ├── __init__.py                     # experimental 模块入口
│   ├── config.py                       # 配置管理
│   ├── dispatcher.py                   # 智能调度器
│   ├── metadata.py                     # 元数据管理
│   ├── exceptions.py                   # 自定义异常
│   │
│   ├── generated/                      # 自动生成算子区
│   │   ├── __init__.py
│   │   ├── _metadata.json             # 算子元数据索引
│   │   ├── _template.py               # 代码生成模板
│   │   ├── blas/                      # BLAS 类算子
│   │   │   ├── __init__.py
│   │   │   ├── gemm.py
│   │   │   └── ...
│   │   ├── reduction/                 # 归约算子
│   │   │   ├── __init__.py
│   │   │   ├── sum.py
│   │   │   └── ...
│   │   └── pointwise/                 # 逐元素算子
│   │       ├── __init__.py
│   │       ├── gelu.py
│   │       └── ...
│   │
│   ├── custom/                        # 手写实验性算子
│   │   ├── __init__.py
│   │   └── ...
│   │
│   ├── graduation/                    # 毕业管理模块
│   │   ├── __init__.py
│   │   ├── criteria.py               # 毕业标准检查
│   │   ├── checker.py                # 自动化检查工具
│   │   ├── tracker.json              # 算子状态追踪
│   │   └── reporter.py               # 报告生成器
│   │
│   ├── testing/                      # 实验性算子测试工具
│   │   ├── __init__.py
│   │   ├── accuracy.py               # 精度验证
│   │   ├── performance.py            # 性能测试
│   │   └── report_template.md        # 报告模板
│   │
│   ├── tests/                        # 新增：实验性算子测试（自动生成）
│   │   ├── __init__.py
│   │   ├── test_generated_ops.py
│   │   ├── test_dispatcher.py
│   │   ├── test_graduation.py
│   │   └── benchmarks/
│   │       └── bench_experimental.py
│   │
│   └── tools/                        # 新增：实验性算子管理工具
│       ├── __init__.py
│       ├── import_from_json.py       # JSON 导入工具
│       ├── exp_cli.py                # CLI 工具入口
│       ├── list_ops.py               # 列出算子
│       ├── check_graduation.py       # 检查毕业资格
│       ├── propose_graduation.py     # 生成毕业 PR
│       └── benchmark_runner.py       # 性能测试运行器
│
├── testing/                           # 主测试框架（已存在，稳定内容）
└── utils/                             # 工具函数（已存在）

tests/                                 # 主测试目录（稳定内容）
├── test_*.py                          # 稳定算子的测试
└── ...

tools/                                 # 主工具目录（稳定内容）
├── *.py                               # 稳定的工具脚本
└── ...

docs/
├── experimental/                      # 新增：实验性功能文档
│   ├── README.md                     # 概述
│   ├── user_guide.md                 # 用户指南
│   ├── contributor_guide.md          # 贡献者指南
│   ├── api_reference.md              # API 文档
│   └── graduation_guide.md           # 毕业流程指南

.github/
└── workflows/
    └── experimental_ci.yml            # 新增：实验性算子 CI
```

### 2. 数据流设计

```
┌─────────────────────────────────────────────────────────────┐
│                    用户调用层                                  │
│  flag_gems.experimental.generated.fast_gelu(x)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ExperimentalDispatcher                          │
│  1. 解析输入 (shape, dtype, device)                           │
│  2. 查询元数据索引                                             │
│  3. 匹配最优实现                                               │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │Shape特化│  │通用实现│  │Fallback│
   │  实现   │  │        │  │到稳定版│
   └────────┘  └────────┘  └────────┘
        │            │            │
        └────────────┼────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  性能监控与日志                                │
│  记录执行时间、内存使用、选择的实现                              │
└─────────────────────────────────────────────────────────────┘
```

### 3. 状态机设计（算子生命周期）

```
   ┌──────────┐
   │  新建PR   │
   └─────┬────┘
         │
         ▼
   ┌──────────────┐
   │ EXPERIMENTAL │ ◄────┐
   │  (初始状态)   │      │
   └─────┬────────┘      │
         │               │
         │ 通过基础测试   │ 性能/质量
         ▼               │ 不达标
   ┌──────────────┐      │
   │   VALIDATED  │──────┘
   │  (已验证)     │
   └─────┬────────┘
         │
         │ 满足毕业标准
         ▼
   ┌──────────────┐
   │ GRADUATION_  │
   │  CANDIDATE   │
   │ (毕业候选)    │
   └─────┬────────┘
         │
         │ 通过人工审核
         ▼
   ┌──────────────┐
   │   GRADUATED  │
   │  (已毕业)     │
   └─────┬────────┘
         │
         │ 移动到 ops/
         ▼
   ┌──────────────┐
   │    STABLE    │
   │   (稳定版)    │
   └──────────────┘
         │
         │ 1个版本后
         ▼
   ┌──────────────┐
   │ DEPRECATED   │
   │ (experimental │
   │   别名移除)   │
   └──────────────┘
```

---

## 模块详细设计

### Module 1: 元数据管理 (metadata.py)

#### 1.1 元数据结构定义

```python
# 数据结构（Python TypedDict/dataclass）

OpMetadata:
    - op_name: str                    # 算子名称
    - op_id: str                      # 唯一标识符 (uuid)
    - category: Literal["blas", "reduction", "pointwise", "custom"]
    - status: OpStatus                # 状态枚举
    -
    generation_info:
        - generator_tool: str         # 如 "KernelGen"
        - generator_version: str      # 如 "v0.1.2"
        - source_template: str        # 模板文件路径
        - generation_date: datetime
        - generation_config: dict     # 生成时的配置参数

    validation_info:
        - accuracy_tests:
            - passed: bool
            - test_cases: int
            - last_run: datetime
            - coverage: dict          # {dtype: [shapes]}
        - performance_tests:
            - baselines: List[PerformanceBaseline]
            - last_run: datetime

    hardware_support:
        - tested_devices: List[str]   # ["NVIDIA-A100", "Cambricon-MLU370"]
        - optimal_devices: List[str]
        - shape_constraints: dict      # 形状约束

    graduation_tracking:
        - eligible: bool
        - in_experimental_since: str   # 版本号
        - checklist: GraduationChecklist
        - blocking_issues: List[str]

    code_location:
        - file_path: str
        - line_start: int
        - line_end: int
        - commit_hash: str

PerformanceBaseline:
    - device: str
    - shape: tuple
    - dtype: str
    - speedup: float                  # 相对于参考实现
    - memory_overhead: float          # MB
    - reference_impl: str             # "torch" or "flag_gems.ops"
    - timestamp: datetime

GraduationChecklist:
    - stable_period_met: bool         # 存在 >= 2 版本
    - accuracy_passed: bool           # 精度测试通过
    - performance_passed: bool        # 性能达标
    - multi_hardware_validated: bool  # 多硬件验证
    - code_review_approved: bool      # 代码审核通过
    - documentation_complete: bool    # 文档完整
```

#### 1.2 元数据管理 API

```python
# 功能函数列表

class MetadataManager:
    def __init__(index_file: str)

    def register_op(metadata: OpMetadata) -> None
        # 注册新算子

    def update_op(op_id: str, updates: dict) -> None
        # 更新算子元数据

    def get_op(op_id: str) -> OpMetadata
        # 查询算子元数据

    def query_ops(filters: dict) -> List[OpMetadata]
        # 条件查询：如 {"status": "VALIDATED", "category": "blas"}

    def update_validation_status(op_id: str, test_results: dict) -> None
        # 更新验证状态

    def check_graduation_eligibility(op_id: str) -> GraduationReport
        # 检查毕业资格

    def export_report(op_ids: List[str], format: str) -> str
        # 导出报告 (markdown/json/html)
```

#### 1.3 元数据存储格式

```json
// _metadata.json 示例
{
  "version": "1.0",
  "last_updated": "2025-12-19T10:30:00Z",
  "ops": {
    "fast_gelu_v1": {
      "op_name": "fast_gelu",
      "op_id": "550e8400-e29b-41d4-a716-446655440000",
      "category": "pointwise",
      "status": "VALIDATED",
      "generation_info": {
        "generator_tool": "TritonAutotune",
        "generator_version": "v0.2.1",
        "source_template": "templates/pointwise.py.jinja",
        "generation_date": "2025-12-01T08:00:00Z",
        "generation_config": {
          "block_size": 1024,
          "num_warps": 4
        }
      },
      "validation_info": {
        "accuracy_tests": {
          "passed": true,
          "test_cases": 45,
          "last_run": "2025-12-15T14:22:00Z",
          "coverage": {
            "float32": ["(1024,)", "(256, 256)", "(8, 128, 128)"],
            "float16": ["(1024,)", "(256, 256)"]
          }
        },
        "performance_tests": {
          "baselines": [
            {
              "device": "NVIDIA-A100",
              "shape": [8192, 4096],
              "dtype": "float16",
              "speedup": 1.23,
              "memory_overhead": 0.5,
              "reference_impl": "torch",
              "timestamp": "2025-12-10T10:00:00Z"
            }
          ],
          "last_run": "2025-12-15T14:30:00Z"
        }
      },
      "hardware_support": {
        "tested_devices": ["NVIDIA-A100", "NVIDIA-H800"],
        "optimal_devices": ["NVIDIA-A100"],
        "shape_constraints": {
          "min_size": 256,
          "alignment": 32
        }
      },
      "graduation_tracking": {
        "eligible": false,
        "in_experimental_since": "v4.1",
        "checklist": {
          "stable_period_met": false,
          "accuracy_passed": true,
          "performance_passed": true,
          "multi_hardware_validated": false,
          "code_review_approved": false,
          "documentation_complete": true
        },
        "blocking_issues": [
          "Need validation on Cambricon MLU",
          "Code review pending"
        ]
      },
      "code_location": {
        "file_path": "src/flag_gems/experimental/generated/pointwise/gelu.py",
        "line_start": 15,
        "line_end": 87,
        "commit_hash": "a1b2c3d4"
      }
    }
  }
}
```

---

### Module 2: 智能调度器 (dispatcher.py)

#### 2.1 调度器核心逻辑

```python
class ExperimentalDispatcher:
    """
    智能调度器：根据输入特征选择最优算子实现
    """

    def __init__(
        metadata_manager: MetadataManager,
        fallback_strategy: str = "safe",  # "safe", "aggressive", "off"
        enable_profiling: bool = True
    )

    def dispatch(
        op_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        主调度函数

        流程：
        1. 提取输入特征 (shape, dtype, device)
        2. 查询可用实现
        3. 根据策略选择实现
        4. 执行并记录性能
        5. 失败时降级
        """

    def _extract_features(args, kwargs) -> InputFeatures
        # 从输入提取关键特征

    def _find_candidates(op_name: str, features: InputFeatures) -> List[OpCandidate]
        # 查找所有可用的实现

    def _select_best(candidates: List[OpCandidate]) -> OpCandidate
        # 选择最优实现（基于历史性能数据）

    def _execute_with_fallback(candidate: OpCandidate, *args, **kwargs) -> Any
        # 执行算子，失败时降级

    def _record_performance(op_name: str, execution_info: dict) -> None
        # 记录性能数据到元数据

# 辅助数据结构
InputFeatures:
    - shapes: List[tuple]
    - dtypes: List[torch.dtype]
    - device: torch.device
    - contiguous: bool

OpCandidate:
    - op_id: str
    - implementation: callable
    - priority: int
    - expected_speedup: float
```

#### 2.2 Fallback 策略设计

```python
# 降级顺序（可配置）

Strategy "safe":
    1. experimental.generated.shape_specialized  # 形状特化版本
    2. experimental.generated.general           # 通用版本
    3. flag_gems.ops.stable                     # 稳定版
    4. torch.native                             # PyTorch 原生

Strategy "aggressive":
    1. experimental.generated.shape_specialized
    2. experimental.generated.general
    3. torch.native                             # 跳过稳定版，直接原生

Strategy "off":
    - 不降级，失败即报错
```

#### 2.3 性能缓存机制

```python
class PerformanceCache:
    """
    缓存历史性能数据，用于调度决策
    """

    cache_structure:
        {
            (op_name, device, shape_signature, dtype): {
                "implementations": {
                    "exp_v1": {"avg_time": 0.5, "std": 0.02, "samples": 100},
                    "exp_v2": {"avg_time": 0.45, "std": 0.03, "samples": 80},
                    "stable": {"avg_time": 0.6, "std": 0.01, "samples": 50}
                },
                "best_implementation": "exp_v2"
            }
        }

    def get_best_impl(key) -> str
    def update_perf(key, impl, time) -> None
    def invalidate(op_name) -> None  # 算子更新时清除缓存
```

---

### Module 3: 毕业管理 (graduation/)

#### 3.1 毕业标准检查器 (criteria.py)

```python
class GraduationCriteria:
    """
    定义和检查毕业标准
    """

    # 标准定义
    STABLE_PERIOD_VERSIONS = 2
    MIN_TEST_COVERAGE = 20  # 最少测试的 shape 数量
    MIN_HARDWARE_COUNT = 2
    MIN_SPEEDUP = 0.8

    def check_stable_period(op_metadata: OpMetadata) -> CheckResult
        """检查是否在 experimental 存在足够版本"""

    def check_accuracy(op_metadata: OpMetadata) -> CheckResult
        """检查精度测试覆盖率和通过率"""

    def check_performance(op_metadata: OpMetadata) -> CheckResult
        """检查性能达标情况"""

    def check_multi_hardware(op_metadata: OpMetadata) -> CheckResult
        """检查多硬件验证"""

    def check_code_quality(op_metadata: OpMetadata) -> CheckResult
        """检查代码质量（需要人工审核标记）"""

    def check_all(op_metadata: OpMetadata) -> GraduationReport
        """执行所有检查并生成报告"""

CheckResult:
    - passed: bool
    - score: float       # 0.0 - 1.0
    - details: str
    - recommendations: List[str]
```

#### 3.2 自动化检查工具 (checker.py)

```python
class AutomatedChecker:
    """
    定期运行的自动化检查工具
    """

    def scan_all_ops() -> List[str]
        """扫描所有实验性算子"""

    def check_graduation_eligibility_batch(op_ids: List[str]) -> dict
        """批量检查毕业资格"""

    def generate_graduation_candidates() -> List[str]
        """生成毕业候选列表"""

    def create_graduation_issue(op_id: str) -> str
        """在 GitHub 创建毕业追踪 issue"""

    def update_tracker(results: dict) -> None
        """更新 tracker.json"""
```

#### 3.3 毕业提案生成器 (propose_graduation.py)

```python
def generate_graduation_pr(op_id: str) -> dict:
    """
    为满足条件的算子生成毕业 PR

    步骤：
    1. 验证毕业资格
    2. 生成文件移动 patch
    3. 创建别名和 DeprecationWarning
    4. 更新 __init__.py 导入
    5. 生成 PR 描述（包含性能报告）
    6. 创建 git branch 和 commit

    返回：
        {
            "branch": "graduation/fast_gelu_v1",
            "pr_title": "[Graduation] Promote fast_gelu to stable ops",
            "pr_body": "...",
            "files_changed": [...]
        }
    """
```

#### 3.4 状态追踪器 (tracker.json)

```json
{
  "tracking_version": "1.0",
  "last_updated": "2025-12-19T10:30:00Z",
  "graduation_candidates": [
    {
      "op_id": "550e8400-e29b-41d4-a716-446655440000",
      "op_name": "fast_gelu",
      "status": "READY_FOR_REVIEW",
      "eligible_since": "2025-12-15",
      "graduation_score": 0.95,
      "blocking_issues": [],
      "pr_number": null,
      "next_action": "Create graduation PR"
    }
  ],
  "recently_graduated": [
    {
      "op_name": "efficient_softmax",
      "graduated_version": "v4.2",
      "graduation_date": "2025-12-10",
      "deprecation_date": "v4.3"
    }
  ]
}
```

---

### Module 4: 配置管理 (config.py)

```python
class ExperimentalConfig:
    """
    实验性功能全局配置
    """

    # 调度策略
    dispatch_strategy: str = "safe"

    # 性能监控
    enable_profiling: bool = True
    profiling_warmup: int = 3
    profiling_repeat: int = 10

    # Fallback 行为
    fallback_on_error: bool = True
    fallback_on_slow: bool = False
    slow_threshold: float = 1.2  # 比参考慢 20% 时降级

    # 警告设置
    show_warnings: bool = True
    warning_level: str = "INFO"  # "DEBUG", "INFO", "WARNING"

    # 缓存设置
    enable_perf_cache: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 秒

    # 日志设置
    log_file: Optional[str] = None
    log_level: str = "INFO"

    @classmethod
    def from_env() -> ExperimentalConfig
        """从环境变量加载配置"""

    @classmethod
    def from_file(path: str) -> ExperimentalConfig
        """从配置文件加载"""

# 环境变量支持
ENV_PREFIX = "FLAGGEMS_EXP_"
# 例如: FLAGGEMS_EXP_DISPATCH_STRATEGY=aggressive
```

---

### Module 5: API 层 (experimental/__init__.py)

```python
"""
FlagGems Experimental Ops

Usage:
    import flag_gems.experimental as fg_exp

    # 使用生成的算子
    result = fg_exp.generated.fast_gelu(x)

    # 使用自定义实验算子
    result = fg_exp.custom.my_op(x, y)

    # 全局启用
    fg_exp.enable(groups=['generated'], fallback=True)
"""

from .dispatcher import ExperimentalDispatcher
from .metadata import MetadataManager
from .config import ExperimentalConfig

# 全局实例
_config = ExperimentalConfig.from_env()
_metadata_mgr = MetadataManager("experimental/generated/_metadata.json")
_dispatcher = ExperimentalDispatcher(_metadata_mgr, _config.dispatch_strategy)

# 子模块
from . import generated
from . import custom
from . import graduation

def enable(
    groups: Optional[List[str]] = None,
    fallback: bool = True,
    warning: bool = True,
    **config_overrides
) -> None:
    """
    全局启用实验性算子

    Args:
        groups: 启用的组 ['generated', 'custom'] 或 None (全部)
        fallback: 失败时是否降级
        warning: 是否显示警告信息
        **config_overrides: 配置覆盖
    """

def disable() -> None:
    """禁用实验性算子"""

def get_config() -> ExperimentalConfig:
    """获取当前配置"""

def set_config(config: ExperimentalConfig) -> None:
    """设置配置"""

def list_ops(
    category: Optional[str] = None,
    status: Optional[str] = None
) -> List[dict]:
    """列出所有实验性算子"""

def get_op_info(op_name: str) -> dict:
    """获取算子详细信息"""

__all__ = [
    'enable',
    'disable',
    'get_config',
    'set_config',
    'list_ops',
    'get_op_info',
    'generated',
    'custom',
    'graduation',
]
```

---

## 开发任务清单

### Phase 1: 基础设施 (Estimated: 2 weeks)

#### Task 1.1: 创建目录结构
- [ ] 创建 `src/flag_gems/experimental/` 目录及子目录
- [ ] 创建 `tests/experimental/` 测试目录
- [ ] 创建 `tools/experimental/` 工具目录
- [ ] 创建 `docs/experimental/` 文档目录
- [ ] 添加所有必要的 `__init__.py` 文件

**文件清单**:
```
src/flag_gems/experimental/__init__.py
src/flag_gems/experimental/config.py
src/flag_gems/experimental/dispatcher.py
src/flag_gems/experimental/metadata.py
src/flag_gems/experimental/exceptions.py
src/flag_gems/experimental/generated/__init__.py
src/flag_gems/experimental/generated/_metadata.json
src/flag_gems/experimental/generated/_template.py
src/flag_gems/experimental/custom/__init__.py
src/flag_gems/experimental/graduation/__init__.py
src/flag_gems/experimental/graduation/criteria.py
src/flag_gems/experimental/graduation/checker.py
src/flag_gems/experimental/graduation/tracker.json
src/flag_gems/experimental/graduation/reporter.py
src/flag_gems/experimental/testing/__init__.py
src/flag_gems/experimental/testing/accuracy.py
src/flag_gems/experimental/testing/performance.py
src/flag_gems/experimental/testing/report_template.md
```

**验收标准**:
- 所有目录和文件创建完成
- 所有 `__init__.py` 包含基本的 docstring
- 能够成功 `import flag_gems.experimental`

---

#### Task 1.2: 实现元数据管理模块
- [ ] 定义元数据数据结构（TypedDict 或 dataclass）
- [ ] 实现 `MetadataManager` 类
  - [ ] `register_op()` 注册算子
  - [ ] `update_op()` 更新元数据
  - [ ] `get_op()` 查询单个算子
  - [ ] `query_ops()` 条件查询
  - [ ] `update_validation_status()` 更新验证状态
  - [ ] `check_graduation_eligibility()` 检查毕业资格
  - [ ] `export_report()` 导出报告
- [ ] 实现 JSON 序列化/反序列化
- [ ] 添加元数据版本管理
- [ ] 实现并发安全（文件锁）

**主要文件**:
- `src/flag_gems/experimental/metadata.py` (约 400-500 行)

**依赖**:
- Python standard library: json, datetime, uuid, threading
- 可选: pydantic (数据验证)

**验收标准**:
- 所有 API 函数实现完成
- 单元测试覆盖率 > 90%
- 支持并发读写
- 能够处理损坏的 JSON 文件

---

#### Task 1.3: 实现配置管理模块
- [ ] 定义 `ExperimentalConfig` dataclass
- [ ] 实现环境变量加载 `from_env()`
- [ ] 实现配置文件加载 `from_file()`
- [ ] 实现配置验证
- [ ] 添加配置热更新支持

**主要文件**:
- `src/flag_gems/experimental/config.py` (约 200 行)

**配置项**:
```python
# 必需配置
- dispatch_strategy
- fallback_on_error

# 可选配置
- enable_profiling
- profiling_warmup
- profiling_repeat
- fallback_on_slow
- slow_threshold
- show_warnings
- warning_level
- enable_perf_cache
- cache_size
- cache_ttl
- log_file
- log_level
```

**验收标准**:
- 支持环境变量和配置文件两种加载方式
- 配置验证能捕获非法值
- 有合理的默认值

---

#### Task 1.4: 实现自定义异常
- [ ] 定义异常层次结构
- [ ] 实现各类异常

**主要文件**:
- `src/flag_gems/experimental/exceptions.py` (约 100 行)

**异常类型**:
```python
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

**验收标准**:
- 所有异常都有清晰的 docstring
- 异常包含有用的上下文信息
- 支持异常链（cause）

---

### Phase 2: 核心功能 (Estimated: 3 weeks)

#### Task 2.1: 实现智能调度器
- [ ] 实现 `ExperimentalDispatcher` 类核心逻辑
  - [ ] `dispatch()` 主入口
  - [ ] `_extract_features()` 特征提取
  - [ ] `_find_candidates()` 查找候选实现
  - [ ] `_select_best()` 选择最优实现
  - [ ] `_execute_with_fallback()` 执行与降级
  - [ ] `_record_performance()` 性能记录
- [ ] 实现 Fallback 策略
  - [ ] Safe 模式
  - [ ] Aggressive 模式
  - [ ] Off 模式
- [ ] 实现性能缓存 `PerformanceCache`
- [ ] 集成元数据查询

**主要文件**:
- `src/flag_gems/experimental/dispatcher.py` (约 500-600 行)

**关键算法**:
```python
# 形状签名生成（用于缓存键）
def shape_signature(shape: tuple) -> str:
    """
    将形状归一化为签名，相似形状使用同一签名
    例如: (1024, 512) -> "1K_512"
          (1000, 500) -> "1K_512"  (向上取整到最近的 2^n)
    """

# 最优实现选择（基于历史数据）
def select_best_implementation(candidates, perf_cache):
    """
    考虑因素:
    1. 平均执行时间
    2. 性能稳定性（标准差）
    3. 样本数量（置信度）
    4. 最近失败率
    """
```

**验收标准**:
- 能正确调度到实验性算子
- Fallback 机制工作正常
- 性能缓存能加速调度决策
- 异常处理完善

---

#### Task 2.2: 实现测试工具
- [ ] 精度验证工具 `accuracy.py`
  - [ ] 与 torch 参考实现对比
  - [ ] 支持多种 dtype 和 shape
  - [ ] 生成精度报告
- [ ] 性能测试工具 `performance.py`
  - [ ] Benchmark runner
  - [ ] 性能对比（vs torch, vs stable ops）
  - [ ] 内存使用分析
  - [ ] 生成性能报告
- [ ] 报告模板 `report_template.md`

**主要文件**:
- `src/flag_gems/experimental/testing/accuracy.py` (约 300 行)
- `src/flag_gems/experimental/testing/performance.py` (约 400 行)
- `src/flag_gems/experimental/testing/report_template.md`

**精度验证功能**:
```python
def validate_accuracy(
    op_impl: callable,
    reference_impl: callable,
    test_configs: List[TestConfig],
    tolerance: dict = {"atol": 1e-5, "rtol": 1e-3}
) -> AccuracyReport

TestConfig:
    - shape: tuple
    - dtype: torch.dtype
    - device: str
    - additional_args: dict
```

**性能测试功能**:
```python
def benchmark_op(
    op_impl: callable,
    baseline_impls: dict,
    test_configs: List[TestConfig],
    warmup: int = 3,
    repeat: int = 10
) -> PerformanceReport
```

**验收标准**:
- 精度验证准确可靠
- 性能测试结果可复现
- 报告格式清晰易读
- 支持批量测试

---

#### Task 2.3: 集成到 CI/CD
- [ ] 创建 GitHub Actions workflow
  - [ ] 实验性算子精度测试
  - [ ] 性能回归测试
  - [ ] 元数据完整性检查
  - [ ] 代码质量检查
- [ ] 实现 PR 自动评论（性能报告）
- [ ] 添加毕业资格检查
- [ ] 集成到现有 CI pipeline

**主要文件**:
- `.github/workflows/experimental_ci.yml`
- `.github/workflows/graduation_check.yml`

**CI 流程**:
```yaml
# experimental_ci.yml

name: Experimental Ops CI

on:
  pull_request:
    paths:
      - 'src/flag_gems/experimental/**'

jobs:
  accuracy_test:
    # 运行精度测试

  performance_test:
    # 运行性能测试
    # 生成对比报告
    # 自动评论到 PR

  metadata_check:
    # 验证元数据完整性
    # 检查元数据格式

  code_quality:
    # 代码风格检查
    # 静态分析
```

**验收标准**:
- CI 能自动触发
- 测试失败能阻止合并
- PR 自动收到性能报告评论
- CI 运行时间合理（< 30 分钟）

---

### Phase 3: 毕业机制 (Estimated: 2 weeks)

#### Task 3.1: 实现毕业标准检查器
- [ ] 实现 `GraduationCriteria` 类
  - [ ] `check_stable_period()` 版本检查
  - [ ] `check_accuracy()` 精度检查
  - [ ] `check_performance()` 性能检查
  - [ ] `check_multi_hardware()` 多硬件检查
  - [ ] `check_code_quality()` 代码质量检查
  - [ ] `check_all()` 综合检查
- [ ] 定义评分系统
- [ ] 生成详细报告

**主要文件**:
- `src/flag_gems/experimental/graduation/criteria.py` (约 400 行)

**评分系统**:
```python
GraduationScore:
    - stable_period: 20%
    - accuracy: 30%
    - performance: 25%
    - multi_hardware: 15%
    - code_quality: 10%

总分 >= 0.85 才能毕业
```

**验收标准**:
- 所有检查项实现完成
- 评分系统合理
- 报告内容详细清晰
- 能识别所有阻塞问题

---

#### Task 3.2: 实现自动化检查工具
- [ ] 实现定期扫描脚本
- [ ] 批量检查毕业资格
- [ ] 自动更新 `tracker.json`
- [ ] 创建 GitHub Issue 追踪

**主要文件**:
- `src/flag_gems/experimental/graduation/checker.py` (约 300 行)
- `tools/experimental/run_graduation_check.py` (约 150 行)

**定时任务**:
```bash
# 每周运行一次
# .github/workflows/weekly_graduation_check.yml
```

**验收标准**:
- 能自动识别毕业候选
- tracker.json 自动更新
- 自动创建 Issue 并分配给维护者

---

#### Task 3.3: 实现毕业提案生成器
- [ ] 文件移动逻辑
- [ ] 别名生成（带 DeprecationWarning）
- [ ] 更新 `__init__.py` 导入
- [ ] 生成 PR 描述
- [ ] 创建 git branch 和 commit

**主要文件**:
- `src/flag_gems/experimental/graduation/proposer.py` (约 400 行)
- `tools/experimental/propose_graduation.py` (CLI wrapper, 约 100 行)

**PR 模板**:
```markdown
## [Graduation] Promote {op_name} to stable ops

### Summary
This PR promotes the experimental operator `{op_name}` to stable ops.

### Graduation Checklist
- [x] Stable period: {versions} versions
- [x] Accuracy tests: {test_count} tests passed
- [x] Performance: {speedup}x speedup on {devices}
- [x] Multi-hardware: Validated on {device_list}
- [x] Code review: Approved by {reviewers}

### Performance Report
{performance_table}

### Files Changed
- Moved: `experimental/generated/{category}/{file}` → `ops/{file}`
- Updated: `experimental/generated/__init__.py` (added deprecation alias)
- Updated: `flag_gems/__init__.py` (added stable import)
- Updated: `experimental/generated/_metadata.json` (marked as graduated)

### Testing
All existing tests pass. New tests added in `tests/ops/test_{op_name}.py`.

### Migration Guide
For users currently using:
```python
import flag_gems.experimental as fg_exp
fg_exp.generated.{op_name}(x)
```

After this PR, prefer:
```python
import flag_gems
flag_gems.{op_name}(x)  # Stable version
```

The experimental alias will be removed in v{next_version}.
```

**验收标准**:
- 能生成完整的 PR
- 所有文件变更正确
- 别名正确包含 DeprecationWarning
- PR 描述完整清晰

---

### Phase 4: 工具和 CLI (Estimated: 1 week)

#### Task 4.1: 实现 CLI 工具
- [ ] `list` - 列出所有实验性算子
- [ ] `info` - 查看算子详细信息
- [ ] `benchmark` - 运行性能测试
- [ ] `check-graduation` - 检查毕业资格
- [ ] `propose-graduation` - 生成毕业 PR

**主要文件**:
- `tools/experimental/exp_cli.py` (约 500 行)

**CLI 接口**:
```bash
# 安装后可用
flag-gems-exp --help

# 子命令
flag-gems-exp list [--category=blas] [--status=validated]
flag-gems-exp info fast_gelu
flag-gems-exp benchmark fast_gelu --device=cuda:0
flag-gems-exp check-graduation fast_gelu
flag-gems-exp propose-graduation fast_gelu --create-pr
```

**实现方式**:
- 使用 `argparse` 或 `click` 库
- 集成前面实现的所有模块
- 输出格式友好（支持 JSON/表格/Markdown）

**验收标准**:
- 所有子命令实现完成
- 输出格式清晰易读
- 错误处理完善
- 有详细的帮助信息

---

#### Task 4.2: 实现代码生成模板
- [ ] 创建算子代码模板
- [ ] 实现模板填充逻辑
- [ ] 自动生成元数据注释

**主要文件**:
- `src/flag_gems/experimental/generated/_template.py`
- `tools/experimental/codegen.py` (约 300 行)

**模板示例**:
```python
# _template.py

OPERATOR_TEMPLATE = '''
"""
{op_name} - Experimental Implementation

Metadata:
    generator_tool: {generator_tool}
    generator_version: {generator_version}
    generation_date: {generation_date}
    op_id: {op_id}
"""

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

@libentry()
@triton.jit
def {op_name}_kernel(
    {kernel_params}
):
    """
    {kernel_docstring}
    """
    {kernel_body}

def {op_name}({op_params}):
    """
    {op_docstring}

    Args:
        {op_args_doc}

    Returns:
        {op_returns_doc}
    """
    {op_body}
'''

def generate_op(config: dict) -> str:
    """根据配置生成算子代码"""
```

**验收标准**:
- 生成的代码符合 FlagGems 规范
- 元数据注释完整
- 代码格式化良好（通过 black/ruff）

---

### Phase 5: 文档和示例 (Estimated: 1 week)

#### Task 5.1: 编写用户文档
- [ ] README.md - 概述
- [ ] user_guide.md - 用户指南
- [ ] api_reference.md - API 文档
- [ ] faq.md - 常见问题

**主要文件**:
- `docs/experimental/README.md`
- `docs/experimental/user_guide.md`
- `docs/experimental/api_reference.md`
- `docs/experimental/faq.md`

**文档内容**:

`README.md`:
- 什么是 experimental ops
- 为什么使用
- 快速开始
- 与稳定版的区别

`user_guide.md`:
- 安装和配置
- 基本用法
- 高级用法
- 性能调优
- 故障排查

`api_reference.md`:
- 所有公开 API 的详细文档
- 参数说明
- 返回值
- 示例代码

`faq.md`:
- 常见问题解答
- 最佳实践
- 性能优化建议

**验收标准**:
- 文档完整覆盖所有功能
- 包含足够的示例代码
- 格式统一，易于阅读
- 定期更新维护

---

#### Task 5.2: 编写贡献者文档
- [ ] contributor_guide.md - 贡献指南
- [ ] graduation_guide.md - 毕业流程指南
- [ ] codegen_guide.md - 代码生成指南

**主要文件**:
- `docs/experimental/contributor_guide.md`
- `docs/experimental/graduation_guide.md`
- `docs/experimental/codegen_guide.md`

**文档内容**:

`contributor_guide.md`:
- 如何贡献实验性算子
- PR 要求和检查清单
- 代码规范
- 测试要求
- 审核流程

`graduation_guide.md`:
- 毕业标准详解
- 如何准备毕业
- 毕业流程步骤
- 常见阻塞问题及解决方案

`codegen_guide.md`:
- 如何使用代码生成工具
- 模板定制
- 元数据管理
- 最佳实践

**验收标准**:
- 贡献者能根据文档完成贡献
- 流程清晰明确
- 包含实际案例

---

#### Task 5.3: 创建示例代码
- [ ] 基础使用示例
- [ ] 高级功能示例
- [ ] 完整的算子贡献示例
- [ ] 毕业流程示例

**主要文件**:
- `examples/experimental/` 目录

**示例清单**:
```
examples/experimental/
├── 01_basic_usage.py           # 基础使用
├── 02_custom_config.py         # 自定义配置
├── 03_performance_profiling.py # 性能分析
├── 04_fallback_strategy.py     # Fallback 策略
├── 05_contribute_op.py         # 贡献算子
└── 06_graduation_process.py    # 毕业流程
```

**验收标准**:
- 所有示例代码能正常运行
- 包含详细注释
- 覆盖主要使用场景

---

### Phase 6: 测试和质量保障 (贯穿所有阶段)

#### Task 6.1: 单元测试
- [ ] `test_metadata.py` - 元数据管理测试
- [ ] `test_dispatcher.py` - 调度器测试
- [ ] `test_config.py` - 配置管理测试
- [ ] `test_graduation.py` - 毕业机制测试
- [ ] `test_cli.py` - CLI 工具测试

**目标覆盖率**: >= 85%

**主要文件**:
```
tests/experimental/
├── __init__.py
├── test_metadata.py          (约 400 行)
├── test_dispatcher.py        (约 500 行)
├── test_config.py            (约 200 行)
├── test_graduation.py        (约 300 行)
├── test_cli.py               (约 300 行)
├── test_accuracy_tools.py    (约 200 行)
└── test_performance_tools.py (约 200 行)
```

**验收标准**:
- 单元测试覆盖率 >= 85%
- 所有边界条件测试
- 异常处理测试
- 并发安全测试

---

#### Task 6.2: 集成测试
- [ ] 端到端流程测试
- [ ] 多算子协同测试
- [ ] Fallback 机制测试
- [ ] 毕业流程完整测试

**主要文件**:
```
tests/experimental/integration/
├── test_e2e_generated_op.py
├── test_e2e_graduation.py
└── test_multi_device.py
```

**验收标准**:
- 覆盖关键业务流程
- 能发现模块间集成问题

---

#### Task 6.3: 性能测试
- [ ] Benchmark suite
- [ ] 性能回归测试
- [ ] 内存泄漏检测

**主要文件**:
```
tests/experimental/benchmarks/
├── bench_dispatcher_overhead.py
├── bench_cache_performance.py
└── bench_ops_comparison.py
```

**验收标准**:
- 调度器开销 < 100μs
- 缓存命中率 > 80%
- 无内存泄漏

---

## 实施计划

### 时间线（总计约 9 周）

```
Week 1-2:  Phase 1 - 基础设施
  ├─ Week 1: Task 1.1, 1.2 (目录结构 + 元数据管理)
  └─ Week 2: Task 1.3, 1.4 (配置管理 + 异常处理)

Week 3-5:  Phase 2 - 核心功能
  ├─ Week 3: Task 2.1 (智能调度器)
  ├─ Week 4: Task 2.2 (测试工具)
  └─ Week 5: Task 2.3 (CI/CD 集成)

Week 6-7:  Phase 3 - 毕业机制
  ├─ Week 6: Task 3.1, 3.2 (标准检查 + 自动化工具)
  └─ Week 7: Task 3.3 (毕业提案生成器)

Week 8:    Phase 4 - 工具和 CLI
  ├─ Task 4.1 (CLI 工具)
  └─ Task 4.2 (代码生成模板)

Week 9:    Phase 5 - 文档和示例
  ├─ Task 5.1 (用户文档)
  ├─ Task 5.2 (贡献者文档)
  └─ Task 5.3 (示例代码)

持续:      Phase 6 - 测试和质量保障
  └─ 贯穿所有阶段，每个模块完成后立即编写测试
```

### 里程碑

**Milestone 1 (Week 2)**: 基础设施完成
- 目录结构创建
- 元数据管理和配置管理可用
- 能够注册和查询实验性算子

**Milestone 2 (Week 5)**: 核心功能完成
- 调度器正常工作
- 测试工具可用
- CI 集成完成
- 能够提交第一个实验性算子 PR

**Milestone 3 (Week 7)**: 毕业机制完成
- 毕业标准检查器工作
- 自动化检查定期运行
- 能够生成毕业 PR

**Milestone 4 (Week 9)**: 项目完成
- 所有工具和文档完成
- 测试覆盖率达标
- 准备发布和推广

### 人员分工建议

**角色 1: 核心开发者** (1-2 人)
- 负责 Phase 1, 2, 3
- 核心模块开发
- 架构设计和技术决策

**角色 2: 测试工程师** (1 人)
- 负责 Phase 6
- 单元测试和集成测试
- CI/CD 配置

**角色 3: 工具开发者** (1 人)
- 负责 Phase 4
- CLI 工具和代码生成
- 自动化脚本

**角色 4: 文档工程师** (1 人)
- 负责 Phase 5
- 所有文档编写
- 示例代码

---

## 技术规范

### 代码规范

#### Python 代码风格
- 遵循 PEP 8
- 使用 `black` 格式化（line-length=100）
- 使用 `ruff` 进行 linting
- 类型注解: 所有公开 API 必须有类型注解

#### 命名规范
- 模块名: `snake_case`
- 类名: `PascalCase`
- 函数名: `snake_case`
- 常量: `UPPER_SNAKE_CASE`
- 私有成员: `_leading_underscore`

#### Docstring 规范
使用 Google Style:
```python
def example_function(param1: int, param2: str) -> bool:
    """Brief description.

    Detailed description (optional).

    Args:
        param1: Description of param1.
        param2: Description of param2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When param1 is negative.

    Example:
        >>> example_function(42, "hello")
        True
    """
```

#### 注释规范
- 代码应自解释，避免冗余注释
- 复杂逻辑必须添加注释
- TODO/FIXME/NOTE 使用标准格式:
  ```python
  # TODO(username): Description of what needs to be done
  # FIXME(username): Description of the bug
  # NOTE: Important information
  ```

### 文件组织规范

#### 导入顺序
1. 标准库
2. 第三方库
3. 本地模块

使用 `isort` 自动排序。

#### 文件头部
```python
# Copyright (c) 2025 FlagGems Contributors
# SPDX-License-Identifier: Apache-2.0

"""
Module description.

This module provides...
"""

from __future__ import annotations  # Python 3.9+

import ...
```

### 测试规范

#### 测试文件命名
- 单元测试: `test_<module_name>.py`
- 集成测试: `test_integration_<feature>.py`
- 性能测试: `bench_<feature>.py`

#### 测试函数命名
```python
def test_<function_name>_<scenario>():
    """Test <function_name> when <scenario>."""
```

#### 测试结构（AAA Pattern）
```python
def test_example():
    # Arrange
    setup_data = ...

    # Act
    result = function_under_test(setup_data)

    # Assert
    assert result == expected_value
```

#### 使用 Fixtures
```python
import pytest

@pytest.fixture
def sample_metadata():
    return MetadataManager(":memory:")

def test_register_op(sample_metadata):
    # Use fixture
    sample_metadata.register_op(...)
```

### 版本控制规范

#### 分支策略
- `master`: 稳定主分支
- `feature/<feature-name>`: 功能分支
- `bugfix/<bug-description>`: 修复分支
- `graduation/<op-name>`: 毕业 PR 分支

#### Commit 消息格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

类型:
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档
- `style`: 格式化
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

示例:
```
feat(experimental): implement ExperimentalDispatcher

Add smart dispatcher with fallback mechanism and performance caching.

Closes #123
```

### 性能要求

#### 调度器性能
- 调度开销 < 100μs (warm cache)
- 调度开销 < 500μs (cold cache)
- 缓存命中率 > 80% (典型工作负载)

#### 元数据操作
- 查询单个算子 < 1ms
- 条件查询 < 10ms
- 更新元数据 < 5ms (包含文件 I/O)

#### 内存使用
- 元数据内存占用 < 100MB (1000 个算子)
- 性能缓存 < 50MB (1000 条记录)

### 安全规范

#### 输入验证
- 所有外部输入必须验证
- 文件路径必须规范化并检查越界
- JSON 数据必须 schema 验证

#### 错误处理
- 不暴露内部实现细节
- 敏感信息不记录到日志
- 异常包含足够的上下文

#### 并发安全
- 文件操作使用锁
- 共享数据结构使用线程安全容器
- 避免全局可变状态

---

## 测试策略

### 测试金字塔

```
        ┌─────────────┐
        │  E2E Tests  │  (10%)
        │   (少量)     │
        ├─────────────┤
        │Integration  │  (20%)
        │   Tests     │
        ├─────────────┤
        │    Unit     │  (70%)
        │    Tests    │
        └─────────────┘
```

### 单元测试清单

#### 元数据管理 (test_metadata.py)
- [ ] 注册新算子
- [ ] 更新已有算子
- [ ] 查询单个算子
- [ ] 条件查询多个算子
- [ ] 更新验证状态
- [ ] 检查毕业资格
- [ ] 导出报告（各种格式）
- [ ] 并发写入安全性
- [ ] 损坏文件恢复
- [ ] 元数据版本迁移

#### 调度器 (test_dispatcher.py)
- [ ] 基本调度流程
- [ ] 特征提取正确性
- [ ] 候选实现查找
- [ ] 最优实现选择
- [ ] Fallback 各种策略
- [ ] 性能缓存工作
- [ ] 性能记录准确性
- [ ] 并发调度安全性
- [ ] 异常处理和降级
- [ ] 配置热更新

#### 配置管理 (test_config.py)
- [ ] 默认配置正确
- [ ] 环境变量加载
- [ ] 配置文件加载
- [ ] 配置验证
- [ ] 非法值拒绝
- [ ] 配置序列化

#### 毕业机制 (test_graduation.py)
- [ ] 稳定期检查
- [ ] 精度检查
- [ ] 性能检查
- [ ] 多硬件检查
- [ ] 代码质量检查
- [ ] 综合评分
- [ ] 报告生成
- [ ] 自动化扫描
- [ ] Issue 创建
- [ ] PR 生成

### 集成测试清单

#### 端到端流程
- [ ] 贡献新算子完整流程
  - 提交 PR → CI 检查 → 合并 → 元数据注册
- [ ] 调度器完整流程
  - 用户调用 → 调度 → 执行 → 性能记录
- [ ] 毕业完整流程
  - 资格检查 → 提案生成 → PR 创建 → 合并 → 别名设置

#### 多模块协同
- [ ] 调度器 + 元数据管理
- [ ] 测试工具 + 元数据管理
- [ ] 毕业检查器 + 所有模块

### 性能测试清单

- [ ] 调度器开销基准测试
- [ ] 缓存性能测试
- [ ] 元数据查询性能
- [ ] 大规模算子场景（1000+ 算子）
- [ ] 内存泄漏检测
- [ ] 并发性能测试

### 兼容性测试

- [ ] PyTorch 版本: 2.0, 2.1, 2.2, 2.3, 2.4+
- [ ] Python 版本: 3.8, 3.9, 3.10, 3.11, 3.12
- [ ] 设备: CPU, CUDA, ROCm
- [ ] 操作系统: Linux, Windows, macOS

### 测试数据

#### 使用真实场景数据
- 从主流模型提取典型 shape
  - LLaMA: attention, feedforward shapes
  - ResNet: conv, pool shapes
  - BERT: embedding, attention shapes

#### Edge Cases
- 空张量
- 单元素张量
- 超大张量 (> 1GB)
- 非连续张量
- 各种 stride 模式

---

## 文档和示例

### 用户文档目录

```
docs/experimental/
├── README.md                    # 概述和快速开始
├── installation.md              # 安装指南
├── user_guide.md                # 用户指南
│   ├── Basic Usage
│   ├── Configuration
│   ├── Performance Tuning
│   └── Troubleshooting
├── api_reference.md             # API 参考
│   ├── experimental module
│   ├── generated submodule
│   ├── custom submodule
│   └── graduation submodule
├── faq.md                       # 常见问题
└── migration_guide.md           # 从 experimental 迁移到 stable
```

### 贡献者文档目录

```
docs/experimental/contributing/
├── contributor_guide.md         # 贡献指南
│   ├── Getting Started
│   ├── PR Requirements
│   ├── Code Standards
│   └── Review Process
├── graduation_guide.md          # 毕业流程指南
│   ├── Graduation Criteria
│   ├── Preparation Checklist
│   ├── Graduation Process
│   └── Common Issues
├── codegen_guide.md             # 代码生成指南
│   ├── Using Templates
│   ├── Metadata Management
│   └── Best Practices
└── architecture.md              # 架构设计文档
    ├── Overall Architecture
    ├── Module Design
    ├── Data Flow
    └── Extension Points
```

### 示例代码清单

```
examples/experimental/
├── basic/
│   ├── 01_simple_usage.py       # 最简单的使用示例
│   ├── 02_with_config.py        # 自定义配置
│   └── 03_error_handling.py     # 错误处理
├── advanced/
│   ├── 04_performance_profiling.py  # 性能分析
│   ├── 05_custom_fallback.py        # 自定义降级策略
│   └── 06_multi_device.py           # 多设备使用
├── contributing/
│   ├── 07_add_generated_op.py       # 添加生成算子
│   ├── 08_add_custom_op.py          # 添加自定义算子
│   └── 09_run_tests.py              # 运行测试
└── graduation/
    └── 10_check_and_propose.py      # 检查和提议毕业
```

---

## 风险和缓解措施

### 技术风险

#### 风险 1: 调度器性能开销
**描述**: 智能调度器可能引入不可接受的性能开销

**缓解措施**:
- 实现高效的性能缓存
- 使用 shape 签名减少缓存 miss
- 提供 "aggressive" 模式直接使用实验算子
- 持续性能监控和优化

#### 风险 2: 元数据文件损坏
**描述**: JSON 文件可能因异常中断而损坏

**缓解措施**:
- 使用原子写入（写临时文件 + 重命名）
- 定期备份元数据文件
- 实现损坏检测和恢复机制
- 考虑使用 SQLite 替代 JSON

#### 风险 3: 并发安全问题
**描述**: 多进程/多线程可能导致数据竞争

**缓解措施**:
- 使用文件锁保护关键操作
- 实现乐观锁（版本号）
- 充分的并发测试
- 文档明确说明并发限制

### 流程风险

#### 风险 4: 毕业标准过于严格
**描述**: 标准太高导致算子长期停留在 experimental

**缓解措施**:
- 制定合理的初始标准
- 定期回顾和调整标准
- 提供"部分毕业"机制（如 stable-beta）
- 社区投票决定边缘案例

#### 风险 5: CI 运行时间过长
**描述**: 大量实验算子导致 CI 时间不可接受

**缓解措施**:
- 分层测试：基础测试 vs 完整测试
- 仅测试变更的算子
- 并行运行测试
- 使用缓存加速

### 社区风险

#### 风险 6: 用户误用实验性功能
**描述**: 用户在生产环境使用未成熟算子

**缓解措施**:
- 清晰的命名空间隔离
- 显眼的警告信息
- 详细的文档说明风险
- 提供稳定性评级

#### 风险 7: 代码质量参差不齐
**描述**: 自动生成代码可能质量低下

**缓解措施**:
- 强制的 CI 检查
- 代码 review 必需
- 提供高质量模板
- 定期清理低质量算子

---

## 度量指标

### 开发进度指标

- [ ] 代码完成度: X / Y 行代码
- [ ] 测试覆盖率: X%
- [ ] 文档完成度: X / Y 页
- [ ] 已完成任务: X / Y

### 质量指标

- 单元测试通过率: >= 100%
- 集成测试通过率: >= 100%
- 代码覆盖率: >= 85%
- 静态分析问题: 0
- 文档覆盖率: 100% (所有公开 API)

### 性能指标

- 调度器开销 (warm): < 100μs
- 调度器开销 (cold): < 500μs
- 缓存命中率: > 80%
- 元数据查询: < 1ms
- CI 运行时间: < 30 min

### 使用指标（发布后）

- 实验性算子数量
- 毕业算子数量
- 毕业成功率
- 平均毕业时间
- PR 合并时间
- 社区贡献者数量

---

## 附录

### A. 参考项目

类似机制的参考:
- PyTorch `torch.nn.experimental`
- TensorFlow `tf.experimental`
- Rust `std::experimental`
- Python `__future__` module

### B. 工具依赖

**必需**:
- Python >= 3.8
- PyTorch >= 2.0
- Triton >= 2.0

**开发工具**:
- black (代码格式化)
- ruff (linting)
- pytest (测试)
- pytest-cov (覆盖率)
- mypy (类型检查)

**可选**:
- click (CLI)
- rich (终端输出)
- pydantic (数据验证)
- plotly (性能可视化)

### C. 术语表

- **Experimental Ops**: 实验性算子，未经充分验证的算子实现
- **Stable Ops**: 稳定算子，已通过所有验证的算子实现
- **Dispatcher**: 调度器，根据输入选择最优算子实现
- **Fallback**: 降级，当首选实现失败时使用备选实现
- **Graduation**: 毕业，实验性算子晋升为稳定算子
- **Metadata**: 元数据，描述算子的结构化信息
- **Shape Signature**: 形状签名，将相似形状归一化的标识符
- **Performance Baseline**: 性能基准，参考实现的性能数据

### D. 联系方式

- 项目维护者: [待定]
- 邮件列表: [待定]
- GitHub Issues: https://github.com/FlagOpen/FlagGems/issues
- 讨论区: [待定]

---

## 文档变更历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|----------|
| v1.0 | 2025-12-19 | Claude | 初始版本 |

---

**文档结束**

_此文档为 FlagGems Experimental Ops 实施的完整技术规范。所有开发工作应严格按照本文档执行。_
