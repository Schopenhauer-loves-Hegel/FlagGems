#!/usr/bin/env python3
"""
FlagGems算子数据提取脚本
从FlagGems仓库中提取算子的torch实现、triton实现和测试代码，生成JSON数据集
"""

import os
import re
import json
import ast
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OperatorExtractor:
    def __init__(self, repo_path: str, output_dir: str):
        self.repo_path = Path(repo_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ops_dir = self.repo_path / "src/flag_gems/ops"
        self.tests_dir = self.repo_path / "tests"

        # 算子类型映射
        self.op_type_mapping = {
            'pointwise': ['add', 'sub', 'mul', 'div', 'abs', 'neg', 'relu', 'gelu', 'silu',
                         'exp', 'log', 'sqrt', 'rsqrt', 'sin', 'cos', 'tanh', 'sigmoid',
                         'pow', 'clamp', 'where', 'bitwise_and', 'bitwise_or', 'bitwise_not',
                         'eq', 'ne', 'gt', 'ge', 'lt', 'le', 'isnan', 'isinf', 'isfinite'],
            'reduction': ['sum', 'mean', 'max', 'min', 'all', 'any', 'argmax', 'argmin',
                         'amax', 'amin', 'cumsum', 'cumprod', 'prod', 'var', 'std'],
            'blas': ['mm', 'bmm', 'addmm', 'baddbmm', 'mv', 'outer'],
            'normalization': ['layer_norm', 'batch_norm', 'group_norm', 'rms_norm', 'softmax', 'log_softmax'],
            'conv': ['conv1d', 'conv2d', 'conv_depthwise2d'],
            'attention': ['attention', 'scaled_dot_product_attention'],
            'indexing': ['gather', 'scatter', 'index_select', 'index_add', 'scatter_reduce'],
            'tensor_ops': ['cat', 'stack', 'split', 'chunk', 'reshape', 'transpose', 'permute'],
        }

        # Torch API映射 - 算子名到torch API的映射
        self.torch_api_mapping = self._build_torch_api_mapping()

    def _build_torch_api_mapping(self) -> Dict[str, str]:
        """构建算子名到torch API调用的映射"""
        return {
            # Pointwise ops
            'abs': 'torch.abs(input)',
            'add': 'torch.add(input, other, alpha=alpha)',
            'sub': 'torch.sub(input, other, alpha=alpha)',
            'mul': 'torch.mul(input, other)',
            'div': 'torch.div(input, other, rounding_mode=rounding_mode)',
            'neg': 'torch.neg(input)',
            'relu': 'torch.relu(input)',
            'gelu': 'torch.nn.functional.gelu(input, approximate=approximate)',
            'silu': 'torch.nn.functional.silu(input)',
            'exp': 'torch.exp(input)',
            'log': 'torch.log(input)',
            'sqrt': 'torch.sqrt(input)',
            'rsqrt': 'torch.rsqrt(input)',
            'sin': 'torch.sin(input)',
            'cos': 'torch.cos(input)',
            'tanh': 'torch.tanh(input)',
            'sigmoid': 'torch.sigmoid(input)',
            'pow': 'torch.pow(input, exponent)',
            'clamp': 'torch.clamp(input, min=min, max=max)',
            'where': 'torch.where(condition, x, y)',

            # Comparison ops
            'eq': 'torch.eq(input, other)',
            'ne': 'torch.ne(input, other)',
            'gt': 'torch.gt(input, other)',
            'ge': 'torch.ge(input, other)',
            'lt': 'torch.lt(input, other)',
            'le': 'torch.le(input, other)',

            # Bitwise ops
            'bitwise_and': 'torch.bitwise_and(input, other)',
            'bitwise_or': 'torch.bitwise_or(input, other)',
            'bitwise_not': 'torch.bitwise_not(input)',

            # Check ops
            'isnan': 'torch.isnan(input)',
            'isinf': 'torch.isinf(input)',
            'isfinite': 'torch.isfinite(input)',

            # Reduction ops
            'sum': 'torch.sum(input, dim=dim, keepdim=keepdim)',
            'mean': 'torch.mean(input, dim=dim, keepdim=keepdim)',
            'max': 'torch.max(input, dim=dim, keepdim=keepdim)',
            'min': 'torch.min(input, dim=dim, keepdim=keepdim)',
            'all': 'torch.all(input, dim=dim, keepdim=keepdim)',
            'any': 'torch.any(input, dim=dim, keepdim=keepdim)',
            'argmax': 'torch.argmax(input, dim=dim, keepdim=keepdim)',
            'argmin': 'torch.argmin(input, dim=dim, keepdim=keepdim)',
            'amax': 'torch.amax(input, dim=dim, keepdim=keepdim)',
            'amin': 'torch.amin(input, dim=dim, keepdim=keepdim)',
            'cumsum': 'torch.cumsum(input, dim=dim)',
            'cumprod': 'torch.cumprod(input, dim=dim)',
            'prod': 'torch.prod(input, dim=dim, keepdim=keepdim)',
            'var': 'torch.var(input, dim=dim, keepdim=keepdim, correction=correction)',
            'std': 'torch.std(input, dim=dim, keepdim=keepdim, correction=correction)',

            # BLAS ops
            'mm': 'torch.mm(input, mat2)',
            'bmm': 'torch.bmm(input, mat2)',
            'addmm': 'torch.addmm(bias, input, mat2, beta=beta, alpha=alpha)',
            'baddbmm': 'torch.baddbmm(bias, input, mat2, beta=beta, alpha=alpha)',
            'mv': 'torch.mv(input, vec)',
            'outer': 'torch.outer(input, vec2)',

            # Normalization ops
            'layer_norm': 'torch.nn.functional.layer_norm(input, normalized_shape, weight=weight, bias=bias, eps=eps)',
            'batch_norm': 'torch.nn.functional.batch_norm(input, running_mean, running_var, weight=weight, bias=bias, training=training, momentum=momentum, eps=eps)',
            'group_norm': 'torch.nn.functional.group_norm(input, num_groups, weight=weight, bias=bias, eps=eps)',
            'rms_norm': 'torch.nn.functional.rms_norm(input, normalized_shape, weight=weight, eps=eps)',
            'softmax': 'torch.softmax(input, dim=dim)',
            'log_softmax': 'torch.log_softmax(input, dim=dim)',

            # Conv ops
            'conv1d': 'torch.nn.functional.conv1d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)',
            'conv2d': 'torch.nn.functional.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)',

            # Attention
            'scaled_dot_product_attention': 'torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)',

            # Indexing ops
            'gather': 'torch.gather(input, dim, index)',
            'scatter': 'torch.scatter(input, dim, index, src)',
            'index_select': 'torch.index_select(input, dim, index)',
            'index_add': 'torch.index_add(input, dim, index, source)',

            # Tensor ops
            'cat': 'torch.cat(tensors, dim=dim)',
            'stack': 'torch.stack(tensors, dim=dim)',
            'split': 'torch.split(tensor, split_size_or_sections, dim=dim)',
            'chunk': 'torch.chunk(tensor, chunks, dim=dim)',
            'reshape': 'torch.reshape(input, shape)',
            'transpose': 'torch.transpose(input, dim0, dim1)',
            'permute': 'torch.permute(input, dims)',
        }

    def get_operator_type(self, op_name: str) -> str:
        """根据算子名称推断类型"""
        for op_type, ops in self.op_type_mapping.items():
            if op_name in ops:
                return op_type
        return "general"

    def scan_operators(self) -> List[str]:
        """扫描所有算子文件"""
        op_files = list(self.ops_dir.glob("*.py"))
        # 排除 __init__.py
        op_files = [f for f in op_files if f.stem != "__init__"]
        op_names = [f.stem for f in op_files]
        logger.info(f"Found {len(op_names)} operators")
        return sorted(op_names)

    def read_operator_file(self, op_name: str) -> str:
        """读取算子文件内容"""
        op_file = self.ops_dir / f"{op_name}.py"
        if not op_file.exists():
            logger.warning(f"Operator file not found: {op_file}")
            return ""

        with open(op_file, 'r', encoding='utf-8') as f:
            return f.read()

    def extract_triton_code(self, op_name: str, source_code: str) -> str:
        """提取Triton实现代码"""
        # 简单策略：返回整个文件内容（去掉可能的调试代码）
        # 清理一些不需要的import
        lines = source_code.split('\n')

        # 移除logging相关的行
        filtered_lines = []
        for line in lines:
            if 'logging.debug' in line or 'logging.info' in line:
                continue
            filtered_lines.append(line)

        return '\n'.join(filtered_lines).strip()

    def generate_torch_code(self, op_name: str, source_code: str) -> str:
        """生成对应的torch实现代码"""
        # 从源代码中提取函数签名
        func_signature = self.extract_function_signature(op_name, source_code)

        if not func_signature:
            # 如果无法提取签名，使用默认映射
            if op_name in self.torch_api_mapping:
                torch_call = self.torch_api_mapping[op_name]
                return f"""import torch

def {op_name}({self.get_default_params(op_name)}):
    return {torch_call}"""
            else:
                # 未知算子，返回占位符
                return f"""import torch

def {op_name}(*args, **kwargs):
    # TODO: Implement torch version
    return torch.{op_name}(*args, **kwargs)"""

        # 使用提取的签名生成torch实现
        params_str, call_params = func_signature

        # 生成正确的torch API调用，使用实际参数名
        torch_call = self.generate_torch_api_call(op_name, params_str)

        return f"""import torch

def {op_name}({params_str}):
    return {torch_call}"""

    def extract_function_signature(self, op_name: str, source_code: str) -> Optional[Tuple[str, str]]:
        """从源代码中提取函数签名"""
        # 查找主函数定义（不带下划线后缀的）
        pattern = rf'^def {re.escape(op_name)}\((.*?)\):'

        for line in source_code.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                params = match.group(1)
                # 解析参数
                params_list = [p.strip() for p in params.split(',') if p.strip()]

                # 构建调用参数
                call_params_list = []
                for param in params_list:
                    if '=' in param:
                        # 有默认值的参数
                        param_name = param.split('=')[0].strip()
                        if param_name not in ['*', '**']:
                            call_params_list.append(f"{param_name}={param_name}")
                    elif param == '*':
                        continue
                    elif param.startswith('**'):
                        continue
                    else:
                        # 位置参数
                        call_params_list.append(param)

                return params, ', '.join(call_params_list)

        return None

    def generate_torch_api_call(self, op_name: str, params_str: str) -> str:
        """生成torch API调用，使用实际参数名"""
        # 解析参数
        params = [p.strip() for p in params_str.split(',') if p.strip()]
        param_names = []
        for param in params:
            if '=' in param:
                param_name = param.split('=')[0].strip()
            elif param in ['*', '**kwargs']:
                continue
            else:
                param_name = param.strip()
            if param_name:
                param_names.append(param_name)

        # 根据算子类型和参数生成调用
        op_type = self.get_operator_type(op_name)

        if op_type == 'pointwise':
            if len(param_names) >= 2:
                if 'alpha' in params_str:
                    return f"torch.{op_name}({param_names[0]}, {param_names[1]}, alpha=alpha)"
                elif 'rounding_mode' in params_str:
                    return f"torch.{op_name}({param_names[0]}, {param_names[1]}, rounding_mode=rounding_mode)"
                else:
                    return f"torch.{op_name}({param_names[0]}, {param_names[1]})"
            elif len(param_names) == 1:
                return f"torch.{op_name}({param_names[0]})"
        elif op_type == 'reduction':
            if len(param_names) >= 1:
                if 'dim' in params_str and 'keepdim' in params_str:
                    return f"torch.{op_name}({param_names[0]}, dim=dim, keepdim=keepdim)"
                elif 'dim' in params_str:
                    return f"torch.{op_name}({param_names[0]}, dim=dim)"
                else:
                    return f"torch.{op_name}({param_names[0]})"
        elif op_type == 'blas':
            if 'addmm' in op_name or 'baddbmm' in op_name:
                return f"torch.{op_name}({param_names[0]}, {param_names[1]}, {param_names[2]}, beta=beta, alpha=alpha)" if len(param_names) >= 3 else f"torch.{op_name}(*{param_names})"
            elif len(param_names) >= 2:
                return f"torch.{op_name}({param_names[0]}, {param_names[1]})"
        elif op_type == 'normalization':
            if 'softmax' in op_name:
                if 'dim' in params_str:
                    return f"torch.{op_name}({param_names[0]}, dim=dim)"
                else:
                    return f"torch.{op_name}({param_names[0]})"
            elif 'layer_norm' in op_name:
                return f"torch.nn.functional.layer_norm({', '.join(param_names)})"
            elif 'batch_norm' in op_name:
                return f"torch.nn.functional.batch_norm({', '.join(param_names)})"
        elif op_type == 'indexing':
            if 'gather' in op_name:
                # gather(inp, dim, index, ...) -> torch.gather(inp, dim, index)
                return f"torch.{op_name}({param_names[0]}, {param_names[1]}, {param_names[2]})"
            elif 'scatter' in op_name and len(param_names) >= 4:
                return f"torch.{op_name}({param_names[0]}, {param_names[1]}, {param_names[2]}, {param_names[3]})"

        # 默认：直接传递所有参数
        return f"torch.{op_name}({', '.join(param_names)})"

    def get_default_params(self, op_name: str) -> str:
        """获取默认参数列表"""
        # 根据算子类型返回常见参数
        op_type = self.get_operator_type(op_name)

        if op_type == 'pointwise':
            if 'add' in op_name or 'sub' in op_name:
                return "input, other, *, alpha=1"
            elif 'mul' in op_name or 'div' in op_name:
                return "input, other"
            else:
                return "input"
        elif op_type == 'reduction':
            return "input, dim=None, keepdim=False"
        elif op_type == 'blas':
            if 'mm' in op_name:
                return "input, mat2"
            else:
                return "*args, **kwargs"
        else:
            return "*args, **kwargs"

    def find_test_file(self, op_name: str) -> Optional[Path]:
        """查找包含指定算子测试的文件"""
        test_files = list(self.tests_dir.glob("test_*.py"))

        for test_file in test_files:
            content = test_file.read_text(encoding='utf-8')
            # 查找 @pytest.mark.{op_name} 或 def test_accuracy_{op_name}
            if f"@pytest.mark.{op_name}" in content or f"def test_accuracy_{op_name}" in content:
                return test_file

        return None

    def extract_test_code(self, op_name: str) -> Optional[str]:
        """从测试文件中提取测试代码"""
        test_file = self.find_test_file(op_name)
        if not test_file:
            logger.warning(f"Test file not found for operator: {op_name}")
            return None

        content = test_file.read_text(encoding='utf-8')

        # 查找测试函数
        pattern = rf'@pytest\.mark\.{re.escape(op_name)}.*?(?=(?:@pytest\.mark|def test_|class |$))'
        matches = re.findall(pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        # 尝试直接查找函数定义
        pattern = rf'def test_accuracy_{re.escape(op_name)}\(.*?\):.*?(?=\ndef |\nclass |\n@pytest|$)'
        matches = re.findall(pattern, content, re.DOTALL)

        if matches:
            return matches[0].strip()

        return None

    def convert_test_to_bench_format(self, op_name: str, test_code: str) -> str:
        """将pytest格式的测试转换为bench格式"""
        if not test_code:
            return self.generate_default_test(op_name)

        # 提取参数化装饰器的参数
        parametrize_pattern = r'@pytest\.mark\.parametrize\("([^"]+)",\s*(\[.*?\])\)'
        parametrizes = re.findall(parametrize_pattern, test_code, re.DOTALL)

        # 提取函数签名
        func_pattern = rf'def test_accuracy_{re.escape(op_name)}\((.*?)\):'
        func_match = re.search(func_pattern, test_code)

        if not func_match:
            return self.generate_default_test(op_name)

        params = func_match.group(1)

        # 构建bench格式的装饰器
        decorators = ['import bench',
                      'from bench.sandbox.test.test_parametrize import parametrize, label',
                      'from bench.sandbox.config import DEVICE as device',
                      'from bench.sandbox.utils.accuracy_utils import gems_assert_close as assert_close',
                      'from bench.sandbox.utils.accuracy_utils import to_reference',
                      'import torch',
                      '',
                      f'@label("{op_name}")']

        # 转换参数化装饰器
        for param_names, param_values in parametrizes:
            decorators.append(f'@parametrize("{param_names}", {param_values})')

        # 构建函数体
        func_body = f"def test_{op_name}({params}):"

        # 简化的函数体生成
        return '\n'.join(decorators) + '\n' + func_body + '\n' + self.generate_test_body(op_name, params)

    def generate_test_body(self, op_name: str, params: str) -> str:
        """生成测试函数体"""
        # 根据参数生成测试体
        param_list = [p.strip().split('=')[0] for p in params.split(',') if p.strip()]

        has_shape = 'shape' in param_list
        has_dtype = 'dtype' in param_list

        body_lines = []

        if has_shape and has_dtype:
            body_lines.append('    x = torch.randn(shape, dtype=dtype, device=device)')
            body_lines.append('    ref_x = to_reference(x, True)')
            body_lines.append('')
            body_lines.append(f'    ref_out = bench.{op_name}(ref_x)')
            body_lines.append(f'    res_out = bench.triton.{op_name}(x)')
            body_lines.append('')
            body_lines.append('    assert_close(res_out, ref_out, dtype)')
        else:
            # 默认实现
            body_lines.append('    # TODO: Implement test body')
            body_lines.append('    pass')

        return '\n'.join(body_lines)

    def generate_default_test(self, op_name: str) -> str:
        """生成默认的测试代码"""
        return f"""import bench
from bench.sandbox.test.test_parametrize import parametrize, label
from bench.sandbox.config import DEVICE as device
from bench.sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from bench.sandbox.utils.accuracy_utils import to_reference
import torch

@label("{op_name}")
@parametrize("shape", [(32, 32), (64, 64), (128, 128)])
@parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_{op_name}(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=device)
    ref_x = to_reference(x, True)

    ref_out = bench.{op_name}(ref_x)
    res_out = bench.triton.{op_name}(x)

    assert_close(res_out, ref_out, dtype)"""

    def extract_metadata(self, op_name: str, source_code: str, test_code: Optional[str]) -> Dict:
        """提取算子元数据"""
        metadata = {
            "query": f"操作符名字是 {op_name}",
            "kernel_name": op_name,
            "func_desc": f"The {op_name} operator",
            "func_type": self.get_operator_type(op_name),
            "gpu": "nvidia",
            "input_args": [],
            "output_args": [{"type": "torch.Tensor", "desc": "The output tensor"}]
        }

        # 尝试从函数签名提取参数
        func_sig = self.extract_function_signature(op_name, source_code)
        if func_sig:
            params_str, _ = func_sig
            params = [p.strip() for p in params_str.split(',') if p.strip() and p.strip() not in ['*', '**kwargs']]

            for param in params:
                if '=' in param:
                    param_name, default_value = param.split('=', 1)
                    param_name = param_name.strip()
                    param_type = self.infer_param_type(param_name, default_value.strip())
                else:
                    param_name = param.strip()
                    param_type = self.infer_param_type(param_name)

                metadata["input_args"].append({
                    "name": param_name,
                    "type": param_type,
                    "desc": f"Parameter {param_name}"
                })

        # 增强描述
        metadata["query"] += f"，是一个 {metadata['func_type']} 算子，处理硬件是 Nvidia。"

        return metadata

    def infer_param_type(self, param_name: str, default_value: str = None) -> str:
        """推断参数类型"""
        # 根据参数名和默认值推断类型
        tensor_keywords = ['input', 'tensor', 'mat', 'weight', 'bias', 'other', 'x', 'y']
        int_keywords = ['dim', 'axis', 'size', 'num', 'groups', 'stride', 'padding', 'dilation']
        float_keywords = ['alpha', 'beta', 'eps', 'momentum', 'dropout']
        bool_keywords = ['keepdim', 'training', 'inplace', 'is_causal']

        param_lower = param_name.lower()

        if any(kw in param_lower for kw in tensor_keywords):
            return "torch.Tensor"
        elif any(kw in param_lower for kw in int_keywords):
            return "int"
        elif any(kw in param_lower for kw in float_keywords):
            return "float"
        elif any(kw in param_lower for kw in bool_keywords):
            return "bool"
        elif default_value:
            if default_value.isdigit():
                return "int"
            elif default_value in ['True', 'False']:
                return "bool"
            elif '.' in default_value:
                return "float"

        return "Any"

    def extract_operator(self, op_name: str) -> Optional[Dict]:
        """提取单个算子的完整数据"""
        logger.info(f"Extracting operator: {op_name}")

        # 读取源代码
        source_code = self.read_operator_file(op_name)
        if not source_code:
            return None

        # 提取Triton代码
        triton_code = self.extract_triton_code(op_name, source_code)

        # 生成Torch代码
        torch_code = self.generate_torch_code(op_name, source_code)

        # 提取测试代码
        test_code_raw = self.extract_test_code(op_name)
        test_code = self.convert_test_to_bench_format(op_name, test_code_raw)

        # 提取元数据
        metadata = self.extract_metadata(op_name, source_code, test_code_raw)

        # 组装完整数据
        operator_data = {
            **metadata,
            "triton_kernel_code": triton_code,
            "torch_kernel_code": torch_code,
            "test_func_code": test_code
        }

        return operator_data

    def save_operator_json(self, op_name: str, data: Dict):
        """保存算子数据为JSON文件"""
        output_file = self.output_dir / f"{op_name}.json"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved: {output_file}")

    def extract_all(self):
        """提取所有算子"""
        operators = self.scan_operators()

        success_count = 0
        failed_operators = []

        for op_name in operators:
            try:
                data = self.extract_operator(op_name)
                if data:
                    self.save_operator_json(op_name, data)
                    success_count += 1
                else:
                    failed_operators.append(op_name)
            except Exception as e:
                logger.error(f"Failed to extract {op_name}: {e}")
                failed_operators.append(op_name)

        logger.info(f"\n{'='*60}")
        logger.info(f"Extraction completed!")
        logger.info(f"Success: {success_count}/{len(operators)}")
        if failed_operators:
            logger.info(f"Failed operators: {', '.join(failed_operators)}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Extract operators from FlagGems repository')
    parser.add_argument('--repo-path', type=str,
                       default='/share/project/tj/workspace/FlagGems',
                       help='Path to FlagGems repository')
    parser.add_argument('--output-dir', type=str,
                       default='/share/project/tj/workspace/FlagGems/extracted_operators',
                       help='Output directory for JSON files')

    args = parser.parse_args()

    extractor = OperatorExtractor(args.repo_path, args.output_dir)
    extractor.extract_all()


if __name__ == '__main__':
    main()
