#!/usr/bin/env python3
"""修复已知bug的脚本"""
import json
from pathlib import Path

# Bug #1: 修复类型注解bug
TYPE_ANNOTATION_FIXES = {
    'vdot': {
        'old': 'return torch.vdot(input: Tensor, other: Tensor)',
        'new': 'return torch.vdot(input, other)'
    },
    'resolve_neg': {
        'old': 'return torch.resolve_neg(A: torch.Tensor)',
        'new': 'return torch.resolve_neg(A)'
    },
    'resolve_conj': {
        'old': 'return torch.resolve_conj(A: torch.Tensor)',
        'new': 'return torch.resolve_conj(A)'
    },
}

def fix_type_annotations():
    """修复类型注解bug"""
    print("=== 修复类型注解bug ===\n")

    for op_name, fix in TYPE_ANNOTATION_FIXES.items():
        file_path = Path(f'extracted_operators/{op_name}.json')
        if not file_path.exists():
            print(f"⚠️  {op_name}.json 不存在，跳过")
            continue

        data = json.load(open(file_path))
        old_code = data['torch_kernel_code']

        if fix['old'] in old_code:
            new_code = old_code.replace(fix['old'], fix['new'])
            data['torch_kernel_code'] = new_code

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            print(f"✅ 修复 {op_name}")
        else:
            print(f"⚠️  {op_name} 未找到预期代码")

def validate_fixes():
    """验证修复结果"""
    print("\n=== 验证修复结果 ===\n")

    for op_name in TYPE_ANNOTATION_FIXES.keys():
        file_path = Path(f'extracted_operators/{op_name}.json')
        if not file_path.exists():
            continue

        data = json.load(open(file_path))
        torch_code = data['torch_kernel_code']

        # 检查return语句中是否还有类型注解
        return_lines = [l for l in torch_code.split('\n') if 'return' in l]
        has_annotation = any(':' in line and 'Tensor' in line for line in return_lines)

        if has_annotation:
            print(f"❌ {op_name} 仍有类型注解")
        else:
            print(f"✅ {op_name} 修复成功")

if __name__ == '__main__':
    fix_type_annotations()
    validate_fixes()
    print("\n修复完成！")
