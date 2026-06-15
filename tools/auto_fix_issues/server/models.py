"""Shared Pydantic models for worker and coordinator."""

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class Issue(BaseModel):
    id: str
    operator: str | None = None
    type: str
    test_cmd: str
    benchmark_cmd: str = ""
    severity: str = "unknown"
    hardware: str | None = None
    scope: str = "operator"
    github_issue: int | None = None
    title: str | None = None
    operators: list[str] = Field(default_factory=list)
    files: list[str] = Field(default_factory=list)

    def to_orchestrator_dict(self) -> dict:
        return self.model_dump(exclude_none=False)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    success = "success"
    failed = "failed"
    cancelled = "cancelled"
    needs_review = "needs_review"


class JobResult(BaseModel):
    branch: str | None = None
    branch_url: str | None = None
    test_passed: bool | None = None
    benchmark_passed: bool | None = None
    format_check_passed: bool | None = None
    error_message: str | None = None
    cc_result: dict | None = None


class Job(BaseModel):
    job_id: str
    issue: Issue
    status: JobStatus = JobStatus.queued
    worker_id: str | None = None
    gpu_id: int | None = None
    attempt: int = 0
    branch: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: JobResult | None = None


class JobSubmission(BaseModel):
    issues: list[Issue]


class JobSubmitResponse(BaseModel):
    jobs: list[dict]


class WorkerStatus(BaseModel):
    worker_id: str
    hardware_type: str
    gpu_count: int
    gpus_available: int
    jobs_queued: int
    jobs_running: int
    uptime_seconds: float


class WorkerInfo(BaseModel):
    worker_id: str
    hardware_type: str
    url: str
    gpu_count: int = 0
    gpus_available: int = 0
    jobs_queued: int = 0
    jobs_running: int = 0
    healthy: bool = True
    last_heartbeat: datetime | None = None


class RoutingResult(BaseModel):
    batch_id: str
    routed: list[dict]
    unroutable: list[dict]
