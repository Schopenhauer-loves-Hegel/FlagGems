#!/usr/bin/env python3
"""
清理所有残留的 tle 导入
"""

import re
from pathlib import Path

def process_file(file_path):
    """处理单个文件，移除 tle 导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 移除各种形式的 tle 导入
    patterns = [
        r'from flag_gems\.utils import triton_lang_extension as tle\n',
        r'from \.\.utils import triton_lang_extension as tle\n',
        r'from \.\.\.utils import triton_lang_extension as tle\n',
    ]

    for pattern in patterns:
        content = re.sub(pattern, '', content)

    # 清理多余的空行
    content = re.sub(r'\n\n\n+', '\n\n', content)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("清理残留的 tle 导入")
    print("=" * 60)

    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    triton_files = list(operators_dir.glob("*/*_triton.py"))

    modified_count = 0

    for triton_file in sorted(triton_files):
        op_name = triton_file.parent.name

        # 检查是否有 tle 导入
        with open(triton_file, 'r', encoding='utf-8') as f:
            if 'triton_lang_extension as tle' not in f.read():
                continue

        try:
            if process_file(triton_file):
                print(f"✓ {op_name:30s} - 已清理")
                modified_count += 1
        except Exception as e:
            print(f"✗ {op_name:30s} - 错误: {e}")

    print("\n" + "=" * 60)
    print(f"处理完成: {modified_count} 个文件被修改")
    print("=" * 60)

if __name__ == '__main__':
    main()
