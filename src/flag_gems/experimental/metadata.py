"""
Metadata Management for Experimental Operators

This module provides data structures and management for experimental operator metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import uuid


class OpStatus(str, Enum):
    """Operator lifecycle status"""
    EXPERIMENTAL = "EXPERIMENTAL"
    VALIDATED = "VALIDATED"
    GRADUATION_CANDIDATE = "GRADUATION_CANDIDATE"
    GRADUATED = "GRADUATED"


class OpCategory(str, Enum):
    """Operator category"""
    POINTWISE = "pointwise"
    REDUCTION = "reduction"
    BLAS = "blas"
    CUSTOM = "custom"


@dataclass
class GenerationInfo:
    """Information about how the operator was generated"""
    generator_tool: str = "unknown"
    generator_version: str = "unknown"
    source_template: Optional[str] = None
    generation_date: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationInfo:
    """Validation test results"""
    accuracy_passed: bool = False
    test_cases_total: int = 0
    test_cases_passed: int = 0
    last_run: Optional[str] = None
    coverage: Dict[str, List[Any]] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """Performance benchmark data"""
    device: str
    shape: List[int]
    dtype: str
    speedup: float
    memory_overhead: Optional[float] = None
    reference_impl: str = "torch"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CodeLocation:
    """Location of the operator code"""
    file_path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    commit_hash: Optional[str] = None


@dataclass
class OpMetadata:
    """Complete metadata for an experimental operator"""
    op_name: str
    op_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: OpCategory = OpCategory.CUSTOM
    status: OpStatus = OpStatus.EXPERIMENTAL

    generation_info: GenerationInfo = field(default_factory=GenerationInfo)
    validation_info: ValidationInfo = field(default_factory=ValidationInfo)
    performance_baselines: List[PerformanceBaseline] = field(default_factory=list)
    code_location: Optional[CodeLocation] = None

    hardware_support: Dict[str, Any] = field(default_factory=dict)
    graduation_tracking: Dict[str, Any] = field(default_factory=dict)

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        # Convert enums to strings
        result['category'] = self.category.value
        result['status'] = self.status.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> OpMetadata:
        """Create from dictionary"""
        # Convert string enums back
        if 'category' in data and isinstance(data['category'], str):
            data['category'] = OpCategory(data['category'])
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = OpStatus(data['status'])

        # Reconstruct nested dataclasses
        if 'generation_info' in data and isinstance(data['generation_info'], dict):
            data['generation_info'] = GenerationInfo(**data['generation_info'])
        if 'validation_info' in data and isinstance(data['validation_info'], dict):
            data['validation_info'] = ValidationInfo(**data['validation_info'])
        if 'code_location' in data and isinstance(data['code_location'], dict):
            data['code_location'] = CodeLocation(**data['code_location'])
        if 'performance_baselines' in data:
            data['performance_baselines'] = [
                PerformanceBaseline(**pb) if isinstance(pb, dict) else pb
                for pb in data['performance_baselines']
            ]

        return cls(**data)


class MetadataManager:
    """Manages experimental operator metadata"""

    def __init__(self, metadata_file: str | Path):
        """
        Initialize metadata manager

        Args:
            metadata_file: Path to the _metadata.json file
        """
        self.metadata_file = Path(metadata_file)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Ensure metadata file exists with valid structure"""
        if not self.metadata_file.exists():
            self._save_index({
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "ops": {}
            })

    def _load_index(self) -> Dict[str, Any]:
        """Load the metadata index from file"""
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_index(self, index: Dict[str, Any]):
        """Save the metadata index to file"""
        index['last_updated'] = datetime.now().isoformat()
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def register_op(self, metadata: OpMetadata) -> None:
        """
        Register a new operator

        Args:
            metadata: Operator metadata to register
        """
        index = self._load_index()
        index['ops'][metadata.op_id] = metadata.to_dict()
        self._save_index(index)

    def update_op(self, op_id: str, updates: Dict[str, Any]) -> None:
        """
        Update existing operator metadata

        Args:
            op_id: Operator ID
            updates: Dictionary of fields to update
        """
        index = self._load_index()
        if op_id not in index['ops']:
            raise KeyError(f"Operator {op_id} not found in metadata")

        index['ops'][op_id].update(updates)
        index['ops'][op_id]['updated_at'] = datetime.now().isoformat()
        self._save_index(index)

    def get_op(self, op_id: str) -> OpMetadata:
        """
        Get operator metadata by ID

        Args:
            op_id: Operator ID

        Returns:
            OpMetadata object
        """
        index = self._load_index()
        if op_id not in index['ops']:
            raise KeyError(f"Operator {op_id} not found in metadata")

        return OpMetadata.from_dict(index['ops'][op_id])

    def query_ops(self, filters: Optional[Dict[str, Any]] = None) -> List[OpMetadata]:
        """
        Query operators by filters

        Args:
            filters: Dictionary of field: value filters

        Returns:
            List of matching OpMetadata objects
        """
        index = self._load_index()
        results = []

        for op_data in index['ops'].values():
            if filters:
                match = all(
                    op_data.get(key) == value
                    for key, value in filters.items()
                )
                if match:
                    results.append(OpMetadata.from_dict(op_data))
            else:
                results.append(OpMetadata.from_dict(op_data))

        return results

    def get_op_by_name(self, op_name: str) -> Optional[OpMetadata]:
        """
        Get operator by name (returns first match)

        Args:
            op_name: Operator name

        Returns:
            OpMetadata object or None if not found
        """
        ops = self.query_ops({"op_name": op_name})
        return ops[0] if ops else None

    def list_all_ops(self) -> List[str]:
        """List all operator IDs"""
        index = self._load_index()
        return list(index['ops'].keys())

    def delete_op(self, op_id: str) -> None:
        """
        Delete an operator from metadata

        Args:
            op_id: Operator ID to delete
        """
        index = self._load_index()
        if op_id in index['ops']:
            del index['ops'][op_id]
            self._save_index(index)
