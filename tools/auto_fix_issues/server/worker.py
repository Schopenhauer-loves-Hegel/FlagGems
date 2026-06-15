"""Worker server: manages GPU resources and runs fix pipelines on this machine."""

import asyncio
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import yaml
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse

_parent_dir = os.path.join(os.path.dirname(__file__), "..")
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from device_manager import DeviceManager
from orchestrator import load_config, load_dotenv

from . import runner
from .models import (
    Issue,
    Job,
    JobResult,
    JobStatus,
    JobSubmission,
    JobSubmitResponse,
    WorkerStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------

_WORKER_CONFIG: dict = {}
_ORCH_CONFIG: dict = {}
_DEVICE_MGR: DeviceManager | None = None
_START_TIME: float = 0.0


@dataclass
class RunInfo:
    gpu_id: int
    cancel_event: threading.Event
    future: asyncio.Future | None = None


class WorkerState:
    def __init__(self, state_path: str):
        self.state_path = state_path
        self.jobs: dict[str, Job] = {}
        self.queue: deque[str] = deque()
        self.running: dict[str, RunInfo] = {}
        self._load()

    def _load(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path) as f:
                data = json.load(f)
            for job_data in data.get("jobs", []):
                job = Job.model_validate(job_data)
                if job.status in (JobStatus.running, JobStatus.queued):
                    job.status = JobStatus.failed
                    job.result = JobResult(
                        error_message="Worker restarted while job was active"
                    )
                    job.updated_at = datetime.now(timezone.utc)
                self.jobs[job.job_id] = job
            logger.info(f"Loaded {len(self.jobs)} jobs from state file")
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    def save(self):
        try:
            data = {"jobs": [j.model_dump(mode="json") for j in self.jobs.values()]}
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp, self.state_path)
        except Exception as e:
            logger.warning(f"Failed to save state: {e}")

    def add_job(self, issue: Issue) -> Job:
        job = Job(
            job_id=uuid.uuid4().hex[:12],
            issue=issue,
            status=JobStatus.queued,
            worker_id=_WORKER_CONFIG.get("worker_id", ""),
        )
        self.jobs[job.job_id] = job
        self.queue.append(job.job_id)
        self.save()
        return job

    def update_job(self, job_id: str, **kwargs):
        if job_id not in self.jobs:
            return
        job = self.jobs[job_id]
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = datetime.now(timezone.utc)
        self.save()


_STATE: WorkerState | None = None
_PROCESSING_TASK: asyncio.Task | None = None
_SHUTDOWN = False
_EXECUTOR: ThreadPoolExecutor | None = None


# ---------------------------------------------------------------------------
# Processing loop
# ---------------------------------------------------------------------------


async def _processing_loop():
    global _SHUTDOWN
    max_retries = _ORCH_CONFIG.get("max_retries", 2)
    poll_interval = _ORCH_CONFIG.get("poll_interval", 10)

    while not _SHUTDOWN:
        # Launch queued jobs if GPUs are available
        while _STATE.queue:
            gpu_id = _DEVICE_MGR.acquire()
            if gpu_id is None:
                break

            job_id = _STATE.queue.popleft()
            job = _STATE.jobs.get(job_id)
            if not job or job.status == JobStatus.cancelled:
                _DEVICE_MGR.release(gpu_id)
                continue

            _STATE.update_job(job_id, status=JobStatus.running, gpu_id=gpu_id)

            cancel_event = threading.Event()
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(
                _EXECUTOR,
                _run_issue_wrapper,
                job_id,
                job.issue,
                gpu_id,
                cancel_event,
                job.attempt,
            )
            _STATE.running[job_id] = RunInfo(
                gpu_id=gpu_id,
                cancel_event=cancel_event,
                future=future,
            )

        # Check completed tasks
        for job_id in list(_STATE.running.keys()):
            info = _STATE.running[job_id]
            if info.future and info.future.done():
                _DEVICE_MGR.release(info.gpu_id)
                del _STATE.running[job_id]

                try:
                    result_dict = info.future.result()
                except Exception as e:
                    logger.exception(f"Job {job_id} raised exception")
                    result_dict = {"status": "failed", "error_message": str(e)}

                status = result_dict.get("status", "failed")
                job = _STATE.jobs.get(job_id)

                if status in ("failed",) and job and job.attempt + 1 < max_retries:
                    _STATE.update_job(
                        job_id,
                        status=JobStatus.queued,
                        attempt=job.attempt + 1,
                        result=JobResult(**_extract_result(result_dict)),
                    )
                    _STATE.queue.append(job_id)
                    logger.info(
                        f"Job {job_id} failed, retrying "
                        f"(attempt {job.attempt + 2}/{max_retries})"
                    )
                else:
                    job_status = _map_status(status)
                    _STATE.update_job(
                        job_id,
                        status=job_status,
                        branch=result_dict.get("branch"),
                        result=JobResult(**_extract_result(result_dict)),
                    )
                    logger.info(f"Job {job_id} finished with status={job_status}")

        await asyncio.sleep(poll_interval)


def _run_issue_wrapper(
    job_id: str,
    issue: Issue,
    gpu_id: int,
    cancel_event: threading.Event,
    attempt: int,
) -> dict:
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(
        script_dir, _ORCH_CONFIG.get("template", "templates/fix_issue.md")
    )
    results_dir = os.path.join(
        script_dir, _ORCH_CONFIG.get("results_dir", "results")
    )
    log_dir = os.path.join(results_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    issue_dict = issue.to_orchestrator_dict()
    return runner.run_single_issue(
        issue_dict=issue_dict,
        gpu_id=gpu_id,
        config=_ORCH_CONFIG,
        template_path=template_path,
        log_dir=log_dir,
        cancel_event=cancel_event,
        attempt=attempt,
    )


def _extract_result(d: dict) -> dict:
    return {
        "branch": d.get("branch"),
        "branch_url": d.get("branch_url"),
        "test_passed": d.get("test_passed"),
        "benchmark_passed": d.get("benchmark_passed"),
        "format_check_passed": d.get("format_check_passed"),
        "error_message": d.get("error_message"),
        "cc_result": d.get("cc_result"),
    }


def _map_status(s: str) -> JobStatus:
    mapping = {
        "success": JobStatus.success,
        "failed": JobStatus.failed,
        "cancelled": JobStatus.cancelled,
        "needs_review": JobStatus.needs_review,
    }
    return mapping.get(s, JobStatus.failed)


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def verify_api_key(x_api_key: str | None = Header(None)):
    expected = _WORKER_CONFIG.get("api_key")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def load_worker_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "worker_config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _WORKER_CONFIG, _ORCH_CONFIG, _DEVICE_MGR, _STATE
    global _START_TIME, _PROCESSING_TASK, _SHUTDOWN, _EXECUTOR

    _START_TIME = time.monotonic()
    _SHUTDOWN = False

    config_path = os.environ.get("WORKER_CONFIG")
    _WORKER_CONFIG = load_worker_config(config_path)

    worker_id = _WORKER_CONFIG.get("worker_id")
    if not worker_id:
        import socket
        hw = _WORKER_CONFIG.get("hardware_type", "unknown")
        _WORKER_CONFIG["worker_id"] = f"worker-{hw}-{socket.gethostname()}"

    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(script_dir, ".env"))

    orch_config_path = _WORKER_CONFIG.get(
        "orchestrator_config",
        os.path.join(script_dir, "config.yaml"),
    )
    if not os.path.isabs(orch_config_path):
        orch_config_path = os.path.join(
            os.path.dirname(config_path or os.path.join(os.path.dirname(__file__), "worker_config.yaml")),
            orch_config_path,
        )
    _ORCH_CONFIG = load_config(orch_config_path)

    device_cfg = _ORCH_CONFIG.get("device", {})
    lock_dir = device_cfg.get("lock_dir", "/tmp/auto_fix_gpu_locks")
    gpu_ids = device_cfg.get("gpu_ids")
    _DEVICE_MGR = DeviceManager(lock_dir, gpu_ids)

    state_dir = os.path.join(script_dir, _ORCH_CONFIG.get("results_dir", "results"))
    os.makedirs(state_dir, exist_ok=True)
    _STATE = WorkerState(os.path.join(state_dir, "worker_state.json"))

    _EXECUTOR = ThreadPoolExecutor(
        max_workers=len(_DEVICE_MGR.gpu_ids),
        thread_name_prefix="issue-runner",
    )
    _PROCESSING_TASK = asyncio.create_task(_processing_loop())

    logger.info(
        f"Worker started: id={_WORKER_CONFIG['worker_id']} "
        f"hw={_WORKER_CONFIG.get('hardware_type')} "
        f"gpus={_DEVICE_MGR.gpu_ids}"
    )

    yield

    _SHUTDOWN = True
    if _PROCESSING_TASK:
        _PROCESSING_TASK.cancel()
        try:
            await _PROCESSING_TASK
        except asyncio.CancelledError:
            pass

    for info in _STATE.running.values():
        info.cancel_event.set()
    _EXECUTOR.shutdown(wait=True, cancel_futures=True)
    _DEVICE_MGR.release_all()
    _STATE.save()
    logger.info("Worker shut down")


app = FastAPI(title="FlagGems Auto-Fix Worker", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/status")
async def get_status(_=Depends(verify_api_key)):
    return WorkerStatus(
        worker_id=_WORKER_CONFIG.get("worker_id", ""),
        hardware_type=_WORKER_CONFIG.get("hardware_type", "unknown"),
        gpu_count=len(_DEVICE_MGR.gpu_ids),
        gpus_available=_DEVICE_MGR.available_count(),
        jobs_queued=len(_STATE.queue),
        jobs_running=len(_STATE.running),
        uptime_seconds=time.monotonic() - _START_TIME,
    )


@app.post("/jobs", status_code=202)
async def submit_jobs(body: JobSubmission, _=Depends(verify_api_key)):
    created = []
    for issue in body.issues:
        job = _STATE.add_job(issue)
        created.append({
            "job_id": job.job_id,
            "issue_id": issue.id,
            "status": job.status.value,
        })
    return JobSubmitResponse(jobs=created)


@app.get("/jobs")
async def list_jobs(
    status: str | None = None,
    _=Depends(verify_api_key),
):
    jobs = list(_STATE.jobs.values())
    if status:
        jobs = [j for j in jobs if j.status.value == status]
    return {
        "jobs": [j.model_dump(mode="json") for j in jobs],
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, _=Depends(verify_api_key)):
    job = _STATE.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.model_dump(mode="json")


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, _=Depends(verify_api_key)):
    job = _STATE.jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in (
        JobStatus.success,
        JobStatus.failed,
        JobStatus.cancelled,
        JobStatus.needs_review,
    ):
        raise HTTPException(status_code=409, detail=f"Job already {job.status.value}")

    if job_id in _STATE.running:
        _STATE.running[job_id].cancel_event.set()

    if job_id in _STATE.queue:
        _STATE.queue.remove(job_id)

    _STATE.update_job(job_id, status=JobStatus.cancelled)
    return {"job_id": job_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="FlagGems Auto-Fix Worker Server")
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(__file__), "worker_config.yaml"),
        help="Path to worker_config.yaml",
    )
    parser.add_argument("--host", default=None, help="Override bind host")
    parser.add_argument("--port", type=int, default=None, help="Override bind port")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.environ["WORKER_CONFIG"] = args.config

    wc = load_worker_config(args.config)
    host = args.host or wc.get("host", "0.0.0.0")
    port = args.port or wc.get("port", 8100)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
