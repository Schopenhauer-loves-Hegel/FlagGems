#!/usr/bin/env python3
"""
将JSON数据转换为Python文件
每个算子生成3个文件：
1. {op}_triton.py - Triton实现
2. {op}_torch.py - Torch实现
3. {op}_test.py - 测试代码
"""

import json
from pathlib import Path
import shutil

def convert_json_to_py(json_dir: Path, output_dir: Path):
    """将JSON转换为Python文件"""

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统计信息
    total = 0
    success = 0

    print("=" * 60)
    print("开始转换JSON到Python文件")
    print("=" * 60)

    # 遍历所有JSON文件
    json_files = sorted(json_dir.glob("*.json"))

    for json_file in json_files:
        total += 1
        op_name = json_file.stem

        try:
            # 读取JSON数据
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 创建算子目录
            op_dir = output_dir / op_name
            op_dir.mkdir(exist_ok=True)

            # 1. 生成Triton实现文件
            triton_file = op_dir / f"{op_name}_triton.py"
            with open(triton_file, 'w', encoding='utf-8') as f:
                f.write(f'"""\n')
                f.write(f'Triton实现 - {op_name}\n')
                f.write(f'算子类型: {data.get("func_type", "unknown")}\n')
                f.write(f'描述: {data.get("func_desc", "")}\n')
                f.write(f'"""\n\n')
                f.write(data['triton_kernel_code'])

            # 2. 生成Torch实现文件
            torch_file = op_dir / f"{op_name}_torch.py"
            with open(torch_file, 'w', encoding='utf-8') as f:
                f.write(f'"""\n')
                f.write(f'Torch参考实现 - {op_name}\n')
                f.write(f'算子类型: {data.get("func_type", "unknown")}\n')
                f.write(f'描述: {data.get("func_desc", "")}\n')
                f.write(f'"""\n\n')
                f.write(data['torch_kernel_code'])

            # 3. 生成测试文件
            test_file = op_dir / f"{op_name}_test.py"
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(f'"""\n')
                f.write(f'测试代码 - {op_name}\n')
                f.write(f'算子类型: {data.get("func_type", "unknown")}\n')
                f.write(f'"""\n\n')
                f.write(data['test_func_code'])

            # 4. 生成README文件（包含元数据）
            readme_file = op_dir / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(f'# {op_name}\n\n')
                f.write(f'## 基本信息\n\n')
                f.write(f'- **算子名**: {data.get("kernel_name", op_name)}\n')
                f.write(f'- **算子类型**: {data.get("func_type", "unknown")}\n')
                f.write(f'- **目标硬件**: {data.get("gpu", "nvidia")}\n')
                f.write(f'- **描述**: {data.get("func_desc", "")}\n\n')

                f.write(f'## 查询语句\n\n')
                f.write(f'{data.get("query", "")}\n\n')

                f.write(f'## 输入参数\n\n')
                if data.get('input_args'):
                    f.write('| 参数名 | 类型 | 描述 |\n')
                    f.write('|--------|------|------|\n')
                    for arg in data['input_args']:
                        f.write(f'| {arg.get("name", "")} | {arg.get("type", "")} | {arg.get("desc", "")} |\n')
                else:
                    f.write('无参数信息\n')

                f.write(f'\n## 输出参数\n\n')
                if data.get('output_args'):
                    f.write('| 类型 | 描述 |\n')
                    f.write('|------|------|\n')
                    for arg in data['output_args']:
                        f.write(f'| {arg.get("type", "")} | {arg.get("desc", "")} |\n')
                else:
                    f.write('无输出信息\n')

                f.write(f'\n## 文件说明\n\n')
                f.write(f'- `{op_name}_triton.py` - Triton kernel实现（FlagGems原始代码）\n')
                f.write(f'- `{op_name}_torch.py` - PyTorch参考实现（groundtruth）\n')
                f.write(f'- `{op_name}_test.py` - 测试代码（bench格式）\n')

            success += 1
            print(f"✓ {op_name:20s} → {op_dir}")

        except Exception as e:
            print(f"✗ {op_name:20s} - 错误: {e}")

    print("=" * 60)
    print(f"转换完成: {success}/{total} 成功")
    print(f"输出目录: {output_dir}")
    print("=" * 60)

    # 生成总索引文件
    generate_index(output_dir, json_files)

def generate_index(output_dir: Path, json_files: list):
    """生成索引文件"""
    index_file = output_dir / "INDEX.md"

    with open(index_file, 'w', encoding='utf-8') as f:
        f.write('# 算子代码索引\n\n')
        f.write(f'共 {len(json_files)} 个算子\n\n')

        # 按算子类型分组
        from collections import defaultdict
        by_type = defaultdict(list)

        for json_file in json_files:
            op_name = json_file.stem
            data = json.load(open(json_file))
            op_type = data.get('func_type', 'unknown')
            by_type[op_type].append(op_name)

        f.write('## 按类型索引\n\n')
        for op_type, ops in sorted(by_type.items(), key=lambda x: -len(x[1])):
            f.write(f'### {op_type} ({len(ops)}个)\n\n')
            for op in sorted(ops):
                f.write(f'- [{op}](./{op}/README.md)\n')
            f.write('\n')

        f.write('## 快速导航\n\n')
        f.write('### 常用算子\n\n')
        common_ops = ['add', 'mul', 'matmul', 'softmax', 'layer_norm', 'gather', 'conv2d']
        for op in common_ops:
            if (output_dir / op).exists():
                f.write(f'- [{op}](./{op}/README.md) ')
                f.write(f'[[Triton](./{op}/{op}_triton.py)] ')
                f.write(f'[[Torch](./{op}/{op}_torch.py)] ')
                f.write(f'[[Test](./{op}/{op}_test.py)]\n')

        f.write('\n### 所有算子（字母序）\n\n')
        for json_file in sorted(json_files):
            op_name = json_file.stem
            f.write(f'- [{op_name}](./{op_name}/README.md)\n')

    print(f"\n✓ 索引文件已生成: {index_file}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='将JSON转换为Python文件')
    parser.add_argument('--input-dir', type=str,
                       default='/share/project/tj/workspace/FlagGems/extracted_operators',
                       help='JSON文件输入目录')
    parser.add_argument('--output-dir', type=str,
                       default='/share/project/tj/workspace/FlagGems/operators_py',
                       help='Python文件输出目录')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    convert_json_to_py(input_dir, output_dir)

if __name__ == '__main__':
    main()
