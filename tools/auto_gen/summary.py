"""Summary tracking for auto-gen orchestrator."""

import json
import os
from datetime import datetime, timezone


def utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


class Summary:
    """Manages the summary.json file with real-time updates."""

    def __init__(self, path: str):
        self.path = path
        self.data = {
            "start_time": utc_timestamp(),
            "end_time": None,
            "summary": {
                "total": 0,
                "success": 0,
                "failed": 0,
                "in_progress": 0,
            },
            "operators": {},
        }
        self._save()

    def add_operator(self, operator: str, gpu_id: int, attempt: int):
        """Record that an operator task has started."""
        self.data["operators"][operator] = {
            "status": "in_progress",
            "gpu_id": gpu_id,
            "attempt": attempt,
            "worktree_path": None,
            "branch": None,
            "start_time": utc_timestamp(),
            "end_time": None,
            "duration_seconds": None,
            "accuracy_passed": None,
            "error_message": None,
            "cc_result": None,
        }
        self._recount()
        self._save()

    def update_operator(self, operator: str, **kwargs):
        """Update fields for an operator."""
        if operator in self.data["operators"]:
            self.data["operators"][operator].update(kwargs)
            self._recount()
            self._save()

    def finalize(self):
        """Mark the run as complete."""
        self.data["end_time"] = utc_timestamp()
        self._save()

    def _recount(self):
        """Recount summary statistics."""
        ops = self.data["operators"]
        self.data["summary"]["total"] = len(ops)
        self.data["summary"]["success"] = sum(
            1 for v in ops.values() if v["status"] == "success"
        )
        self.data["summary"]["failed"] = sum(
            1 for v in ops.values() if v["status"] in ("failed", "cancelled")
        )
        self.data["summary"]["in_progress"] = sum(
            1 for v in ops.values() if v["status"] in ("in_progress", "retrying")
        )

    def _save(self):
        """Write summary to disk."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
