#!/usr/bin/env python3
"""
移除 libentry 装饰器及其导入
"""

import re
from pathlib import Path

def process_file(file_path):
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # 1. 移除 libentry 的导入
    # 处理: from ..utils import libentry
    if 'from ..utils import libentry' in content:
        # 检查是否是单独一行导入
        if re.search(r'^from \.\.utils import libentry\s*$', content, re.MULTILINE):
            content = re.sub(r'^from \.\.utils import libentry\s*\n', '', content, flags=re.MULTILINE)
            modified = True
        # 检查是否是多项导入中的一项
        elif re.search(r'from \.\.utils import.*libentry', content):
            # 移除 libentry, 或 , libentry
            content = re.sub(r',\s*libentry\b', '', content)
            content = re.sub(r'\blibentry\s*,\s*', '', content)
            modified = True

    # 2. 移除 @libentry() 装饰器
    # 匹配 @libentry() 单独一行
    if '@libentry()' in content:
        content = re.sub(r'@libentry\(\)\s*\n', '', content)
        modified = True

    # 3. 清理多余的空行
    content = re.sub(r'\n\n\n+', '\n\n', content)

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("移除 libentry 装饰器")
    print("=" * 60)

    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    triton_files = list(operators_dir.glob("*/*_triton.py"))

    modified_count = 0

    for triton_file in sorted(triton_files):
        op_name = triton_file.parent.name

        try:
            if process_file(triton_file):
                print(f"✓ {op_name:30s} - 已处理")
                modified_count += 1
        except Exception as e:
            print(f"✗ {op_name:30s} - 错误: {e}")

    print("\n" + "=" * 60)
    print(f"处理完成: {modified_count} 个文件被修改")
    print("=" * 60)

if __name__ == '__main__':
    main()
