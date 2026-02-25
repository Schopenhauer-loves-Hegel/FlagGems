#!/usr/bin/env python3
"""
修正剩余的 runtime.device 引用
"""

import re
from pathlib import Path

def process_file(file_path):
    """处理单个文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    modified = False

    # 1. 移除错误的 import 语法: from ..runtime import device, torch.cuda
    if 'from ..runtime import device, torch.cuda' in content:
        content = content.replace('from ..runtime import device, torch.cuda', '')
        modified = True

    # 2. 移除 from ..runtime import device
    if 'from ..runtime import device' in content:
        content = content.replace('from ..runtime import device', '')
        modified = True

    # 3. 移除 from ..runtime import backend
    if 'from ..runtime import backend' in content:
        content = content.replace('from ..runtime import backend', '')
        modified = True

    # 4. 替换 device_ = device 为直接的 CUDA 字符串
    if 'device_ = device' in content:
        # 删除这一行
        content = re.sub(r'\ndevice_ = device\n', '\n', content)
        modified = True

    # 5. 替换 device_.name 为 "cuda"
    if 'device_.name' in content:
        content = content.replace('device_.name', '"cuda"')
        modified = True

    # 6. 替换 backend.supports_bfloat16() 为 True (NVIDIA 支持 bfloat16)
    if 'backend.supports_bfloat16()' in content:
        content = content.replace('backend.supports_bfloat16()', 'True')
        modified = True

    # 7. 清理多余的空行
    content = re.sub(r'\n\n\n+', '\n\n', content)

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True

    return False

def main():
    print("=" * 60)
    print("修正剩余的 runtime.device 引用")
    print("=" * 60)

    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    triton_files = list(operators_dir.glob("*/*_triton.py"))

    modified_count = 0

    for triton_file in sorted(triton_files):
        op_name = triton_file.parent.name

        try:
            if process_file(triton_file):
                print(f"✓ {op_name:30s} - 已修正")
                modified_count += 1
        except Exception as e:
            print(f"✗ {op_name:30s} - 错误: {e}")

    print("\n" + "=" * 60)
    print(f"处理完成: {modified_count} 个文件被修改")
    print("=" * 60)

if __name__ == '__main__':
    main()
