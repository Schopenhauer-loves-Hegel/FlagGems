#!/usr/bin/env python3
"""
清理所有 operators_py 中残留的 flag_gems 依赖
包括：tle, runtime, libtuner, libentry, torch_device_fn
"""

import re
from pathlib import Path

def cleanup_file(file_path):
    """清理单个文件"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    original = content
    modified = False
    
    # 1. 替换 tle.program_id → tl.program_id
    if 'tle.program_id' in content:
        content = content.replace('tle.program_id', 'tl.program_id')
        modified = True
    
    # 2. 替换 tle.num_programs → tl.num_programs
    if 'tle.num_programs' in content:
        content = content.replace('tle.num_programs', 'tl.num_programs')
        modified = True
    
    # 3. 移除 runtime 导入（单独一行）
    pattern = r'^from flag_gems import runtime\s*\n'
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, '', content, flags=re.MULTILINE)
        modified = True
    
    # 4. 移除 tle 导入
    pattern = r'^from flag_gems\.utils import triton_lang_extension as tle\s*\n'
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, '', content, flags=re.MULTILINE)
        modified = True
    
    # 5. 移除 torch_device_fn 导入
    pattern = r'^from flag_gems\.runtime import torch_device_fn\s*\n'
    if re.search(pattern, content, re.MULTILINE):
        content = re.sub(pattern, '', content, flags=re.MULTILINE)
        modified = True
    
    # 6. 移除 libentry 导入（从多项导入中）
    content = re.sub(r'\blibentry\s*,\s*', '', content)
    content = re.sub(r',\s*libentry\b', '', content)
    
    # 7. 移除 libtuner 装饰器
    if '@libtuner' in content:
        # 移除 @libtuner(...) 装饰器（可能跨多行）
        pattern = r'@libtuner\([^)]*\)\s*\n'
        content = re.sub(pattern, '', content)
        modified = True
    
    # 8. 移除 libtuner 导入
    content = re.sub(r'\blibtuner\s*,\s*', '', content)
    content = re.sub(r',\s*libtuner\b', '', content)
    
    # 9. 移除 with torch_device_fn.device(...):
    if 'torch_device_fn.device' in content:
        # 找到 with torch_device_fn.device(...): 并移除，调整缩进
        pattern = r'(\s*)with torch_device_fn\.device\([^)]+\):\s*\n'
        
        def fix_indent(match):
            # 移除 with 语句，并减少后续代码的缩进
            return ''
        
        content = re.sub(pattern, fix_indent, content)
        
        # 调整缩进：将后续4个空格的缩进减少
        lines = content.split('\n')
        new_lines = []
        in_with_block = False
        for line in lines:
            # 简单处理：如果行以8个空格开始（原来with块内的代码），减少到4个
            if line.startswith('        ') and not line.startswith('            '):
                new_lines.append(line[4:])  # 减少4个空格
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
        modified = True
    
    # 10. 替换 with torch.cuda.device(...): 为简单的注释或移除
    if 'with torch.cuda.device' in content:
        # 简单处理：移除这个 with 语句，调整缩进
        pattern = r'(\s*)with torch\.cuda\.device\([^)]+\):\s*\n'
        content = re.sub(pattern, '', content)
        
        # 调整缩进
        lines = content.split('\n')
        new_lines = []
        for line in lines:
            if line.startswith('        ') and not line.startswith('            '):
                new_lines.append(line[4:])
            else:
                new_lines.append(line)
        content = '\n'.join(new_lines)
        modified = True
    
    # 11. 清理空白的 import 行
    content = re.sub(r'^from [\w.]+ import\s*$', '', content, flags=re.MULTILINE)
    
    # 12. 清理多余空行
    content = re.sub(r'\n\n\n+', '\n\n', content)
    
    if content != original:
        return content, True
    return content, False

def process_all_files():
    """处理所有 triton 文件"""
    operators_dir = Path("/share/project/tj/workspace/FlagGems/operators_py")
    
    modified_files = []
    
    for op_dir in sorted(operators_dir.iterdir()):
        if not op_dir.is_dir():
            continue
        if op_dir.name in ['common', '__pycache__']:
            continue
        
        triton_file = op_dir / f"{op_dir.name}_triton.py"
        if not triton_file.exists():
            continue
        
        try:
            new_content, modified = cleanup_file(triton_file)
            
            if modified:
                # 备份
                backup = triton_file.with_suffix('.py.bak2')
                if not backup.exists():
                    import shutil
                    shutil.copy(triton_file, backup)
                
                # 写入
                with open(triton_file, 'w') as f:
                    f.write(new_content)
                
                modified_files.append(op_dir.name)
                print(f"✓ {op_dir.name}")
        except Exception as e:
            print(f"✗ {op_dir.name}: {e}")
    
    return modified_files

def main():
    print("="*60)
    print("清理所有残留的 flag_gems 依赖")
    print("="*60)
    print()
    
    modified = process_all_files()
    
    print()
    print("="*60)
    print(f"完成: 修改了 {len(modified)} 个文件")
    print("="*60)
    
    if modified:
        print("\n修改的文件:")
        for name in modified:
            print(f"  - {name}")

if __name__ == '__main__':
    main()
