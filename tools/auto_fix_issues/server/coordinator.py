"""Coordinator server: routes issues to workers by hardware type."""

import asyncio
import logging
import os
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import httpx
import yaml
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse

from .models import (
    Issue,
    JobStatus,
    JobSubmission,
    RoutingResult,
    WorkerInfo,
    WorkerStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config & state
# ---------------------------------------------------------------------------

_CONFIG: dict = {}
_START_TIME: float = 0.0


@dataclass
class WorkerEntry:
    id: str
    url: str
    hardware_type: str
    healthy: bool = True
    fail_count: int = 0
    cached_status: WorkerStatus | None = None
    last_heartbeat: datetime | None = None


@dataclass
class CoordJob:
    coord_job_id: str
    worker_id: str
    worker_job_id: str
    issue: Issue
    status: JobStatus = JobStatus.queued
    result: dict | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CoordinatorState:
    def __init__(self):
        self.workers: dict[str, WorkerEntry] = {}
        self.jobs: dict[str, CoordJob] = {}
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                trust_env=False,
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.aclose()


_STATE = CoordinatorState()
_HEALTH_TASK: asyncio.Task | None = None
_JOB_POLL_TASK: asyncio.Task | None = None
_SHUTDOWN = False


# ---------------------------------------------------------------------------
# Worker health polling
# ---------------------------------------------------------------------------


async def _health_poll_loop():
    interval = _CONFIG.get("poll_interval", 30)
    threshold = _CONFIG.get("unhealthy_threshold", 3)

    while not _SHUTDOWN:
        for w in _STATE.workers.values():
            try:
                headers = _auth_headers_for(w)
                resp = await _STATE.client.get(f"{w.url}/status", headers=headers)
                resp.raise_for_status()
                status = WorkerStatus.model_validate(resp.json())
                w.cached_status = status
                w.last_heartbeat = datetime.now(timezone.utc)
                w.fail_count = 0
                if not w.healthy:
                    logger.info(f"Worker {w.id} recovered")
                w.healthy = True
            except Exception as e:
                w.fail_count += 1
                if w.fail_count >= threshold and w.healthy:
                    w.healthy = False
                    logger.warning(f"Worker {w.id} marked unhealthy: {e}")
        await asyncio.sleep(interval)


# ---------------------------------------------------------------------------
# Job status polling
# ---------------------------------------------------------------------------


async def _job_poll_loop():
    interval = _CONFIG.get("job_poll_interval", 15)

    while not _SHUTDOWN:
        active = defaultdict(list)
        for cj in _STATE.jobs.values():
            if cj.status in (JobStatus.queued, JobStatus.running):
                active[cj.worker_id].append(cj)

        for worker_id, coord_jobs in active.items():
            w = _STATE.workers.get(worker_id)
            if not w or not w.healthy:
                continue
            try:
                headers = _auth_headers_for(w)
                resp = await _STATE.client.get(f"{w.url}/jobs", headers=headers)
                resp.raise_for_status()
                worker_jobs = {j["job_id"]: j for j in resp.json().get("jobs", [])}
                for cj in coord_jobs:
                    wj = worker_jobs.get(cj.worker_job_id)
                    if wj:
                        new_status = wj.get("status", cj.status.value)
                        cj.status = JobStatus(new_status)
                        cj.result = wj.get("result")
                        cj.updated_at = datetime.now(timezone.utc)
            except Exception as e:
                logger.debug(f"Job poll for {worker_id} failed: {e}")

        await asyncio.sleep(interval)


def _auth_headers_for(w: WorkerEntry) -> dict:
    api_key = _CONFIG.get("worker_api_keys", {}).get(w.id)
    if api_key:
        return {"X-API-Key": api_key}
    return {}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def verify_api_key(x_api_key: str | None = Header(None)):
    expected = _CONFIG.get("api_key")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


async def route_issues(issues: list[Issue]) -> RoutingResult:
    by_hw: dict[str | None, list[Issue]] = defaultdict(list)
    for issue in issues:
        by_hw[issue.hardware].append(issue)

    routed = []
    unroutable = []
    batch_id = uuid.uuid4().hex[:12]

    for hw, group in by_hw.items():
        if not hw:
            for issue in group:
                unroutable.append({
                    "issue_id": issue.id,
                    "reason": "hardware field is required",
                })
            continue

        candidates = [
            w for w in _STATE.workers.values()
            if w.hardware_type == hw and w.healthy
        ]
        if not candidates:
            for issue in group:
                unroutable.append({
                    "issue_id": issue.id,
                    "reason": f"no healthy worker for hardware={hw}",
                })
            continue

        # Distribute issues across workers round-robin, sorted by available GPUs
        candidates.sort(
            key=lambda w: w.cached_status.gpus_available if w.cached_status else 0,
            reverse=True,
        )
        per_worker: dict[str, list[Issue]] = defaultdict(list)
        for i, issue in enumerate(group):
            target = candidates[i % len(candidates)]
            per_worker[target.id].append(issue)

        for worker_id, batch in per_worker.items():
            target = _STATE.workers[worker_id]
            try:
                headers = _auth_headers_for(target)
                resp = await _STATE.client.post(
                    f"{target.url}/jobs",
                    json={"issues": [i.model_dump(mode="json") for i in batch]},
                    headers=headers,
                )
                resp.raise_for_status()
                worker_jobs = resp.json().get("jobs", [])

                for wj in worker_jobs:
                    coord_id = uuid.uuid4().hex[:12]
                    issue = next(
                        (i for i in batch if str(i.id) == str(wj["issue_id"])), None
                    )
                    if issue:
                        _STATE.jobs[coord_id] = CoordJob(
                            coord_job_id=coord_id,
                            worker_id=target.id,
                            worker_job_id=wj["job_id"],
                            issue=issue,
                        )
                        routed.append({
                            "job_id": coord_id,
                            "issue_id": wj["issue_id"],
                            "worker_id": target.id,
                            "status": "queued",
                        })
            except Exception as e:
                logger.error(f"Failed to submit to {target.id}: {e}")
                for issue in batch:
                    unroutable.append({
                        "issue_id": issue.id,
                        "reason": f"worker {target.id} submission failed: {e}",
                    })

    return RoutingResult(batch_id=batch_id, routed=routed, unroutable=unroutable)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


def load_coordinator_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "coordinator_config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _CONFIG, _STATE, _START_TIME, _HEALTH_TASK, _JOB_POLL_TASK, _SHUTDOWN

    _START_TIME = time.monotonic()
    _SHUTDOWN = False

    config_path = os.environ.get("COORDINATOR_CONFIG")
    _CONFIG = load_coordinator_config(config_path)

    _STATE.__init__()
    for w_cfg in _CONFIG.get("workers", []):
        entry = WorkerEntry(
            id=w_cfg["id"],
            url=w_cfg["url"].rstrip("/"),
            hardware_type=w_cfg["hardware_type"],
        )
        _STATE.workers[entry.id] = entry

    logger.info(
        f"Coordinator started with {len(_STATE.workers)} workers: "
        f"{[w.id for w in _STATE.workers.values()]}"
    )

    _HEALTH_TASK = asyncio.create_task(_health_poll_loop())
    _JOB_POLL_TASK = asyncio.create_task(_job_poll_loop())

    yield

    _SHUTDOWN = True
    for task in (_HEALTH_TASK, _JOB_POLL_TASK):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    await _STATE.close()
    logger.info("Coordinator shut down")


app = FastAPI(title="FlagGems Auto-Fix Coordinator", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/jobs", status_code=202)
async def submit_jobs(body: JobSubmission, _=Depends(verify_api_key)):
    result = await route_issues(body.issues)
    return result.model_dump(mode="json")


@app.get("/jobs")
async def list_jobs(
    status: str | None = None,
    hardware: str | None = None,
    worker_id: str | None = None,
    _=Depends(verify_api_key),
):
    jobs = list(_STATE.jobs.values())
    if status:
        jobs = [j for j in jobs if j.status.value == status]
    if hardware:
        jobs = [j for j in jobs if j.issue.hardware == hardware]
    if worker_id:
        jobs = [j for j in jobs if j.worker_id == worker_id]

    summary = {
        "total": len(jobs),
        "queued": sum(1 for j in jobs if j.status == JobStatus.queued),
        "running": sum(1 for j in jobs if j.status == JobStatus.running),
        "success": sum(1 for j in jobs if j.status == JobStatus.success),
        "failed": sum(1 for j in jobs if j.status == JobStatus.failed),
    }

    return {
        "jobs": [
            {
                "job_id": j.coord_job_id,
                "issue_id": j.issue.id,
                "worker_id": j.worker_id,
                "hardware": j.issue.hardware,
                "status": j.status.value,
                "result": j.result,
                "created_at": j.created_at.isoformat(),
                "updated_at": j.updated_at.isoformat(),
            }
            for j in jobs
        ],
        "summary": summary,
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, _=Depends(verify_api_key)):
    cj = _STATE.jobs.get(job_id)
    if not cj:
        raise HTTPException(status_code=404, detail="Job not found")

    w = _STATE.workers.get(cj.worker_id)
    if w and w.healthy:
        try:
            headers = _auth_headers_for(w)
            resp = await _STATE.client.get(
                f"{w.url}/jobs/{cj.worker_job_id}", headers=headers
            )
            if resp.status_code == 200:
                worker_data = resp.json()
                cj.status = JobStatus(worker_data.get("status", cj.status.value))
                cj.result = worker_data.get("result")
                cj.updated_at = datetime.now(timezone.utc)
                return {
                    "job_id": cj.coord_job_id,
                    "worker_id": cj.worker_id,
                    "worker_job_id": cj.worker_job_id,
                    "issue": cj.issue.model_dump(mode="json"),
                    "status": cj.status.value,
                    "result": cj.result,
                    "created_at": cj.created_at.isoformat(),
                    "updated_at": cj.updated_at.isoformat(),
                }
        except Exception:
            pass

    return {
        "job_id": cj.coord_job_id,
        "worker_id": cj.worker_id,
        "issue": cj.issue.model_dump(mode="json"),
        "status": cj.status.value,
        "result": cj.result,
        "created_at": cj.created_at.isoformat(),
        "updated_at": cj.updated_at.isoformat(),
        "_note": "status may be stale" if not (w and w.healthy) else None,
    }


@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str, _=Depends(verify_api_key)):
    cj = _STATE.jobs.get(job_id)
    if not cj:
        raise HTTPException(status_code=404, detail="Job not found")

    if cj.status in (
        JobStatus.success,
        JobStatus.failed,
        JobStatus.cancelled,
        JobStatus.needs_review,
    ):
        raise HTTPException(status_code=409, detail=f"Job already {cj.status.value}")

    w = _STATE.workers.get(cj.worker_id)
    if w:
        try:
            headers = _auth_headers_for(w)
            await _STATE.client.delete(
                f"{w.url}/jobs/{cj.worker_job_id}", headers=headers
            )
        except Exception as e:
            logger.warning(f"Failed to cancel on worker: {e}")

    cj.status = JobStatus.cancelled
    cj.updated_at = datetime.now(timezone.utc)
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/workers")
async def list_workers(_=Depends(verify_api_key)):
    result = []
    for w in _STATE.workers.values():
        info = {
            "worker_id": w.id,
            "hardware_type": w.hardware_type,
            "url": w.url,
            "healthy": w.healthy,
            "last_heartbeat": w.last_heartbeat.isoformat() if w.last_heartbeat else None,
        }
        if w.cached_status:
            info.update({
                "gpu_count": w.cached_status.gpu_count,
                "gpus_available": w.cached_status.gpus_available,
                "jobs_queued": w.cached_status.jobs_queued,
                "jobs_running": w.cached_status.jobs_running,
            })
        result.append(info)
    return {"workers": result}


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="FlagGems Auto-Fix Coordinator Server")
    parser.add_argument(
        "-c", "--config",
        default=os.path.join(os.path.dirname(__file__), "coordinator_config.yaml"),
        help="Path to coordinator_config.yaml",
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

    os.environ["COORDINATOR_CONFIG"] = args.config

    cc = load_coordinator_config(args.config)
    host = args.host or cc.get("host", "0.0.0.0")
    port = args.port or cc.get("port", 8000)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
