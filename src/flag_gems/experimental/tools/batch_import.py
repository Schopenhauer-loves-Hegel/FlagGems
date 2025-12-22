#!/usr/bin/env python3
"""
Batch import operators into experimental framework

Usage:
    # Import from filtered results
    python batch_import.py --input selected_batch1.json --batch 1

    # Dry run to preview
    python batch_import.py --input selected_batch1.json --batch 1 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from flag_gems.experimental.tools.import_from_json import JSONImporter
from flag_gems.experimental.metadata import (
    MetadataManager,
    OpMetadata,
    OpCategory,
    OpStatus,
)


class BatchImporter:
    """Batch import operators from filtered results"""

    def __init__(
        self,
        input_file: Path,
        batch: int,
        dry_run: bool = False,
        verbose: bool = True
    ):
        """
        Initialize batch importer

        Args:
            input_file: Path to filtered results JSON
            batch: Batch number (1 or 2)
            dry_run: Preview without making changes
            verbose: Print progress
        """
        self.input_file = Path(input_file)
        self.batch = batch
        self.dry_run = dry_run
        self.verbose = verbose

        # Paths
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.exp_root = Path(__file__).parent.parent
        self.generated_root = self.exp_root / "generated"
        self.metadata_file = self.generated_root / "_metadata.json"

        # Statistics
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }

    def load_filtered_results(self) -> Dict[str, Any]:
        """Load filtered operator results"""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if self.verbose:
            print(f"📂 Loaded filtered results:")
            print(f"   Batch: {data['batch']}")

            # Batch 1 has 'total_operators', Batch 2 has 'selected_operators'
            if 'total_operators' in data:
                print(f"   Total operators: {data['total_operators']}")
                print(f"   Selected: {data['selected_operators']}")
            else:
                print(f"   Total GPT operators: {data.get('total_gpt_operators', 0)}")
                print(f"   New operators: {data.get('new_operators', 0)}")
                print(f"   Selected: {data['selected_operators']}")

        return data

    def check_existing_operators(self) -> set:
        """Get set of already imported operators"""
        metadata_mgr = MetadataManager(self.metadata_file)
        existing = set()

        # Get all operators from metadata
        all_ops = metadata_mgr.query_ops()
        for op_metadata in all_ops:
            existing.add(op_metadata.op_name)

        if self.verbose:
            print(f"\n📋 Already imported: {len(existing)} operators")

        return existing

    def prepare_operator_json(
        self,
        op_name: str,
        op_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare operator data in JSON format for import

        Args:
            op_name: Operator name
            op_data: Operator performance data (from filtered results)

        Returns:
            JSON structure for import_from_json.py
        """
        # Extract code from filtered results
        code = op_data.get('code', '')
        if not code:
            raise ValueError(f"No code found for operator {op_name}")

        # Build JSON structure for import
        json_data = {
            'op_name': op_name,  # Already includes 'aten::' prefix
            'code': code,
            'test_func': '',  # Tests will be generated separately if needed
            'params': {},
            'info': {
                'total': 1,  # Placeholder - actual test info not available yet
                'success': 1,
                'failed': 0
            }
        }

        # Add performance data if available (for batch 1)
        if 'avg_speedup_vs_flaggems' in op_data:
            json_data['performance'] = {
                'batch': self.batch,
                'avg_speedup_vs_flaggems': op_data['avg_speedup_vs_flaggems'],
                'gpt_speedup_vs_cuda': op_data.get('gpt_speedup_vs_cuda'),
                'flaggems_speedup_vs_cuda': op_data.get('flaggems_speedup_vs_cuda')
            }
        # Add performance data for batch 2
        elif 'gpt_speedup_vs_cuda' in op_data:
            json_data['performance'] = {
                'batch': self.batch,
                'gpt_speedup_vs_cuda': op_data['gpt_speedup_vs_cuda']
            }

        return json_data

    def import_operator(
        self,
        op_name: str,
        op_data: Dict[str, Any],
        existing: set
    ) -> bool:
        """
        Import a single operator

        Args:
            op_name: Operator name
            op_data: Operator data
            existing: Set of existing operator names

        Returns:
            True if successful
        """
        # Check if already exists
        if op_name in existing:
            if self.verbose:
                print(f"⏭️  {op_name}: Already imported, skipping")
            self.stats['skipped'] += 1
            return False

        try:
            # Prepare JSON data
            json_data = self.prepare_operator_json(op_name, op_data)

            # Create temporary JSON file
            temp_json = self.project_root / f"temp_{op_name}.json"
            with open(temp_json, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)

            # Import using JSONImporter
            importer = JSONImporter(
                json_file=temp_json,
                category=None,  # Auto-detect
                dry_run=self.dry_run
            )

            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Importing: {op_name}")
                print(f"{'='*60}")

            importer.import_operator()

            # Clean up temp file
            if not self.dry_run:
                temp_json.unlink()

            self.stats['success'] += 1
            return True

        except Exception as e:
            error_msg = f"{op_name}: {str(e)}"
            self.stats['errors'].append(error_msg)
            self.stats['failed'] += 1

            if self.verbose:
                print(f"❌ Error importing {op_name}: {e}")

            return False

    def run(self):
        """Run batch import"""
        print(f"\n{'='*70}")
        print(f"🚀 Batch Import - Batch {self.batch}")
        print(f"{'='*70}\n")

        # Load filtered results
        data = self.load_filtered_results()

        # Check existing operators
        existing = self.check_existing_operators()

        # Get operators to import
        operators = data['operators']
        self.stats['total'] = len(operators)

        print(f"\n📦 Starting import of {self.stats['total']} operators...\n")

        # Import each operator
        for idx, (op_name, op_data) in enumerate(operators.items(), 1):
            print(f"\n[{idx}/{self.stats['total']}] ", end='')
            self.import_operator(op_name, op_data, existing)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print import summary"""
        print(f"\n{'='*70}")
        print(f"📊 Import Summary")
        print(f"{'='*70}")
        print(f"Total:    {self.stats['total']}")
        print(f"Success:  {self.stats['success']} ✅")
        print(f"Skipped:  {self.stats['skipped']} ⏭️")
        print(f"Failed:   {self.stats['failed']} ❌")

        if self.stats['errors']:
            print(f"\n❌ Errors:")
            for error in self.stats['errors']:
                print(f"   - {error}")

        if self.dry_run:
            print(f"\n⚠️  DRY RUN - No files were modified")

        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch import operators from filtered results"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to filtered results JSON"
    )
    parser.add_argument(
        "--batch",
        type=int,
        choices=[1, 2],
        required=True,
        help="Batch number: 1=existing ops, 2=new ops"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making changes"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    try:
        importer = BatchImporter(
            input_file=args.input,
            batch=args.batch,
            dry_run=args.dry_run,
            verbose=not args.quiet
        )
        importer.run()

    except Exception as e:
        print(f"❌ Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
