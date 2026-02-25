# Git Commit

请帮我提交当前的代码更改。

## 提交者信息

- **Author**: gumptao <tj22@tsinghua.org.cn>

## 提交流程

1. 运行 `git status` 查看当前更改
2. 运行 `git diff` 查看具体修改内容
3. 运行 `git log --oneline -5` 参考最近的提交风格
4. 分析更改，生成合适的 commit message
5. 使用 `git add` 添加相关文件（优先添加具体文件，避免 `git add -A`）
6. 使用以下格式提交：

```bash
git commit --author="gumptao <tj22@tsinghua.org.cn>" -m "<commit message>"
```

## 用户补充指导（可选）

$ARGUMENTS

## 注意事项

- 如果用户提供了具体的 commit message，直接使用
- 如果用户提供了补充说明（如 "只提交 src 目录"、"这是一个 bug fix"），根据指导执行
- 如果用户没有提供任何参数，根据更改内容自动生成 commit message
- 遵循项目的 commit 风格（如 `feat:`, `fix:`, `[docs]` 等）
- 不要添加 Co-Authored-By
- 不要 push 到远程，除非用户明确要求
