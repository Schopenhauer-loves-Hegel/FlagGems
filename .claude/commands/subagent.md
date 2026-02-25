# Subagent Task

使用 Task 工具启动一个 subagent 来执行以下任务：

$ARGUMENTS

## 执行规则

- **不指定 model 参数**，让 subagent 继承当前会话的模型
- 使用 `general-purpose` 作为 subagent_type
- 立即调用 Task 工具执行任务
