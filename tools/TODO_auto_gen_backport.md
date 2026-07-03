# auto_gen backport TODO

从 `auto_fix_issues` 迁移回 `auto_gen` 的剩余改进项。

---

## P0 — 基础设施

### 1. 原子 GPU 锁
- **文件**: `device_manager.py` — `acquire()`
- **现状**: `os.path.exists()` 和 `open()` 之间存在 TOCTOU 竞态
- **目标**: 改用 `os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` 原子创建

### 2. 优雅进程终止
- **文件**: `orchestrator.py` — `_kill_cc_process()`
- **现状**: 直接 `SIGKILL`，无等待
- **目标**: `SIGTERM` → 等 10s → `SIGKILL` → 等 5s，给 CC 保存状态的机会

### 3. API 断连自动恢复
- **文件**: `orchestrator.py`
- **现状**: API 连接断开后整个任务从头重来
- **目标**: 检测流式错误 → 提取 session_id → `claude --resume` 接续会话；识别 403/401 直接跳过不重试
- **配置**: 新增 `max_stream_retries`（默认 3）

### 4. Summary 跨次运行保留
- **文件**: `orchestrator.py` — `Summary.__init__()`
- **现状**: 每次运行覆盖 `summary.json`
- **目标**: 加载已有 `summary.json`，只覆盖本次运行的算子，保留历史结果

### 5. 可中断 sleep
- **文件**: `orchestrator.py` — 主循环 sleep
- **现状**: `time.sleep(poll_interval)` 整段阻塞（有 `shutdown_requested` 标志但 sleep 不拆分）
- **目标**: 改为 1s 步进循环 + `shutdown_requested` 检查

### 6. 文件句柄 double-close 防护
- **文件**: `orchestrator.py` — `_kill_cc_process()`、`parse_cc_result()`
- **现状**: 两处都无条件 `close()`，可能 double-close
- **目标**: 加 `if not proc._stdout_file.closed` 检查

---

## P1 — 建议改进

### 7. JSON 解析增强
- **现状**: 简单正则匹配，嵌套 JSON 会失败
- **目标**: 迁移 `_extract_json_object()`（brace-counting + 字符串转义感知）

### 8. `needs_review` 状态
- **现状**: CC 正常退出且 worktree 有改动但输出格式不对时只有 success/failed
- **目标**: 标记为 `needs_review`，避免丢弃可能正确的代码

### 9. Co-Author 清理
- **现状**: commit message 带 `Co-Authored-By`
- **目标**: CC 完成后去掉 commit message 中的 `Co-Authored-By` 行
