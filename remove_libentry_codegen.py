#!/usr/bin/env python3
"""
移除代码生成文件中的 libentry 引用
"""

import re
from pathlib import Path

def process_file(file_path):
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # 1. 移除生成 libentry import 的代码
    patterns = [
        r'code\.writeline\("from flag_gems\.utils import libentry"\)\s*\n',
        r'code\.writeline\("from flag_gems\.utils\.libentry import libentry"\)\s*\n',
    ]

    for pattern in patterns:
        if re.search(pattern, content):
            content = re.sub(pattern, '', content)
            modified = True

    # 2. 移除生成 @libentry() 装饰器的代码
    if 'code.writeline("@libentry()")' in content:
        content = re.sub(r'code\.writeline\("@libentry\(\)"\)\s*\n', '', content)
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
    print("移除代码生成文件中的 libentry 引用")
    print("=" * 60)

    # 需要处理的代码生成文件
    files_to_process = [
        "/share/project/tj/workspace/FlagGems/operators_py/gather/gather_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/scatter/scatter_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/pad/pad_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/tile/tile_triton.py",
        "/share/project/tj/workspace/FlagGems/operators_py/repeat/repeat_triton.py",
    ]

    modified_count = 0

    for file_path in files_to_process:
        file_path_obj = Path(file_path)
        op_name = file_path_obj.parent.name

        try:
            if process_file(file_path):
                print(f"✓ {op_name:30s} - 已修正")
                modified_count += 1
            else:
                print(f"  {op_name:30s} - 无需修改")
        except Exception as e:
            print(f"✗ {op_name:30s} - 错误: {e}")

    print("\n" + "=" * 60)
    print(f"处理完成: {modified_count} 个文件被修改")
    print("=" * 60)

if __name__ == '__main__':
    main()
