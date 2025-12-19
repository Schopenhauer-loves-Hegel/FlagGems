#!/usr/bin/env python3
"""
Import experimental operators from JSON format

This tool imports auto-generated operators from JSON files and integrates
them into the FlagGems experimental framework.

Usage:
    python tools/experimental/import_from_json.py example.json
    python tools/experimental/import_from_json.py example.json --category reduction
    python tools/experimental/import_from_json.py example.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from flag_gems.experimental.metadata import (
    MetadataManager,
    OpMetadata,
    OpCategory,
    OpStatus,
    GenerationInfo,
    ValidationInfo,
    CodeLocation,
)


class JSONImporter:
    """Imports operators from JSON format"""

    def __init__(
        self,
        json_file: Path,
        category: Optional[str] = None,
        dry_run: bool = False
    ):
        """
        Initialize importer

        Args:
            json_file: Path to JSON file containing operator
            category: Force category (pointwise/reduction/blas), auto-detect if None
            dry_run: If True, only show what would be done without making changes
        """
        self.json_file = Path(json_file)
        self.force_category = category
        self.dry_run = dry_run

        # Project paths
        self.project_root = Path(__file__).parent.parent.parent
        self.exp_root = self.project_root / "src" / "flag_gems" / "experimental"
        self.generated_root = self.exp_root / "generated"
        self.tests_root = self.project_root / "tests" / "experimental"
        self.metadata_file = self.generated_root / "_metadata.json"

        # Initialize metadata manager
        self.metadata_mgr = MetadataManager(self.metadata_file)

    def load_json(self) -> Dict[str, Any]:
        """Load and validate JSON file"""
        if not self.json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")

        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate required fields
        required_fields = ['op_name', 'code']
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        return data

    def parse_op_name(self, full_name: str) -> tuple[str, str]:
        """
        Parse operator name from format like 'aten::huber_loss'

        Args:
            full_name: Full operator name

        Returns:
            Tuple of (namespace, op_name)
        """
        if '::' in full_name:
            namespace, op_name = full_name.split('::', 1)
        else:
            namespace = 'aten'
            op_name = full_name

        return namespace, op_name

    def infer_category(self, op_name: str, code: str) -> OpCategory:
        """
        Infer operator category from name and code

        Args:
            op_name: Operator name
            code: Operator code

        Returns:
            OpCategory enum
        """
        if self.force_category:
            return OpCategory(self.force_category)

        # Check for keywords in name
        reduction_keywords = ['sum', 'mean', 'max', 'min', 'prod', 'reduce', 'norm']
        blas_keywords = ['mm', 'matmul', 'gemm', 'bmm', 'dot', 'addmm', 'baddbmm']

        op_lower = op_name.lower()

        if any(kw in op_lower for kw in blas_keywords):
            return OpCategory.BLAS
        elif any(kw in op_lower for kw in reduction_keywords):
            return OpCategory.REDUCTION
        else:
            return OpCategory.POINTWISE

    def extract_metadata_from_json(self, data: Dict[str, Any]) -> OpMetadata:
        """
        Extract operator metadata from JSON data

        Args:
            data: JSON data dictionary

        Returns:
            OpMetadata object
        """
        namespace, op_name = self.parse_op_name(data['op_name'])
        category = self.infer_category(op_name, data['code'])

        # Generation info
        gen_info = GenerationInfo(
            generator_tool="auto_codegen",  # Inferred from JSON structure
            generator_version="unknown",
            generation_date=datetime.now().isoformat(),
            generation_config=data.get('params', {})
        )

        # Validation info
        val_info = ValidationInfo()
        if 'info' in data:
            info = data['info']
            val_info.test_cases_total = info.get('total', 0)
            val_info.test_cases_passed = info.get('success', 0)
            val_info.accuracy_passed = (
                info.get('failed', 0) == 0 and info.get('success', 0) > 0
            )
            val_info.last_run = datetime.now().isoformat()

        # Create metadata
        metadata = OpMetadata(
            op_name=op_name,
            category=category,
            status=OpStatus.EXPERIMENTAL,
            generation_info=gen_info,
            validation_info=val_info
        )

        return metadata

    def generate_code_file(
        self,
        metadata: OpMetadata,
        code: str
    ) -> tuple[Path, str]:
        """
        Generate operator code file with metadata header

        Args:
            metadata: Operator metadata
            code: Operator code

        Returns:
            Tuple of (file_path, file_content)
        """
        # Determine file path
        category_dir = self.generated_root / metadata.category.value
        file_path = category_dir / f"{metadata.op_name}.py"

        # Generate header
        header = f'''"""
{metadata.op_name} - Experimental Implementation

This operator was automatically imported from generated code.

Metadata:
    op_id: {metadata.op_id}
    category: {metadata.category.value}
    status: {metadata.status.value}
    generator_tool: {metadata.generation_info.generator_tool}
    generation_date: {metadata.generation_info.generation_date}

Warning:
    This is an experimental operator and may not be fully optimized or
    tested across all platforms. Use with caution.
"""

'''

        # Combine header and code
        file_content = header + code + "\n"

        return file_path, file_content

    def generate_test_file(
        self,
        metadata: OpMetadata,
        test_code: str
    ) -> tuple[Path, str]:
        """
        Generate test file

        Args:
            metadata: Operator metadata
            test_code: Test code

        Returns:
            Tuple of (file_path, file_content)
        """
        file_path = self.tests_root / f"test_{metadata.op_name}.py"

        # Generate header
        header = f'''"""
Tests for experimental operator: {metadata.op_name}

Auto-imported from generated test code.
Op ID: {metadata.op_id}
"""

'''

        file_content = header + test_code + "\n"

        return file_path, file_content

    def update_init_file(self, metadata: OpMetadata):
        """
        Update __init__.py to export the new operator

        Args:
            metadata: Operator metadata
        """
        category_dir = self.generated_root / metadata.category.value
        init_file = category_dir / "__init__.py"

        # Read current content
        with open(init_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if import already exists
        import_line = f"from .{metadata.op_name} import {metadata.op_name}"
        if import_line in content:
            return  # Already imported

        # Find __all__ and add operator
        if "__all__" in content:
            # Find __all__ = [...]
            all_match = re.search(r'__all__\s*=\s*\[(.*?)\]', content, re.DOTALL)
            if all_match:
                current_exports = all_match.group(1).strip()
                if current_exports:
                    new_exports = f'{current_exports},\n    "{metadata.op_name}"'
                else:
                    new_exports = f'"{metadata.op_name}"'

                new_all = f'__all__ = [\n    {new_exports}\n]'
                content = content[:all_match.start()] + new_all + content[all_match.end():]
        else:
            # Add __all__ if not exists
            content = content.rstrip() + f'\n\n__all__ = [\n    "{metadata.op_name}"\n]\n'

        # Add import statement before __all__
        if "__all__" in content:
            all_pos = content.index("__all__")
            import_section = f"\nfrom .{metadata.op_name} import {metadata.op_name}\n"
            content = content[:all_pos] + import_section + "\n" + content[all_pos:]
        else:
            content = content.rstrip() + f'\n\nfrom .{metadata.op_name} import {metadata.op_name}\n'

        if not self.dry_run:
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write(content)

    def import_operator(self) -> OpMetadata:
        """
        Main import process

        Returns:
            Imported OpMetadata
        """
        print(f"Importing operator from: {self.json_file}")

        # Load JSON
        data = self.load_json()
        print(f"  Loaded JSON with op_name: {data['op_name']}")

        # Extract metadata
        metadata = self.extract_metadata_from_json(data)
        print(f"  Extracted metadata:")
        print(f"    - Op name: {metadata.op_name}")
        print(f"    - Category: {metadata.category.value}")
        print(f"    - Op ID: {metadata.op_id}")
        print(f"    - Test cases: {metadata.validation_info.test_cases_passed}/"
              f"{metadata.validation_info.test_cases_total}")

        # Generate code file
        code_path, code_content = self.generate_code_file(metadata, data['code'])
        print(f"  Generated code file: {code_path.relative_to(self.project_root)}")

        if not self.dry_run:
            code_path.parent.mkdir(parents=True, exist_ok=True)
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code_content)
            print(f"    ✓ Written")

        # Update metadata with code location
        metadata.code_location = CodeLocation(
            file_path=str(code_path.relative_to(self.project_root))
        )

        # Generate test file if test code exists
        if 'test_func' in data and data['test_func']:
            test_path, test_content = self.generate_test_file(
                metadata, data['test_func']
            )
            print(f"  Generated test file: {test_path.relative_to(self.project_root)}")

            if not self.dry_run:
                test_path.parent.mkdir(parents=True, exist_ok=True)
                with open(test_path, 'w', encoding='utf-8') as f:
                    f.write(test_content)
                print(f"    ✓ Written")

        # Update __init__.py
        print(f"  Updating __init__.py for {metadata.category.value}")
        if not self.dry_run:
            self.update_init_file(metadata)
            print(f"    ✓ Updated")

        # Register metadata
        print(f"  Registering metadata")
        if not self.dry_run:
            self.metadata_mgr.register_op(metadata)
            print(f"    ✓ Registered")

        print(f"\n✅ Import completed successfully!")
        if self.dry_run:
            print("   (Dry run - no files were modified)")

        return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Import experimental operators from JSON format"
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to JSON file containing operator"
    )
    parser.add_argument(
        "--category",
        choices=["pointwise", "reduction", "blas"],
        help="Force operator category (auto-detect if not specified)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    try:
        importer = JSONImporter(
            json_file=args.json_file,
            category=args.category,
            dry_run=args.dry_run
        )
        importer.import_operator()
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
