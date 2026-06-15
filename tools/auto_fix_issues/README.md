# auto_fix_issues

Automatically fix FlagGems issues using Claude Code as the coding agent.

## Overview

This tool takes a YAML list of known issues (accuracy failures, runtime errors, etc.) and orchestrates parallel Claude Code sessions — each in its own git worktree — to reproduce, diagnose, and fix the issues.

Each CC session runs with `--dangerously-skip-permissions` (no interactive confirmation) and `--output-format stream-json` for structured logging.

## Quick Start

```bash
# 1. Copy and fill in config
cp config.yaml.example config.yaml
# Edit config.yaml: set flaggems_dir, python_path, device.gpu_ids, etc.

# 2. Copy and fill in .env
cp .env.example .env
# Edit .env: set ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL, and optionally ANTHROPIC_MODEL

# 3. Prepare the issues list
# Edit issues_to_fix.yaml with the issues you want to fix

# 4. Run
python orchestrator.py [-v] [-c config.yaml] [issues_to_fix.yaml]
```

Options:
- `-v` / `--verbose`: Enable debug-level logging
- `-c` / `--config`: Path to config.yaml (default: `config.yaml` in script directory)

## Issues YAML Format

```yaml
issues:
  - id: 418
    operator: sparse_mla_fwd_interface
    type: accuracy_fail          # accuracy_fail | runtime_error | compilation_error | test_error
    severity: major              # optional: major | minor | unknown
    test_cmd: "pytest -m 'sparse_mla_fwd_interface' tests/ --ref cpu -vs"
    benchmark_cmd: "pytest -m 'sparse_mla_fwd_interface' benchmark/ --level core --record log"
```

Required fields: `id`, `operator`, `type`, `test_cmd`

Note: `test_cmd` should be in `pytest ...` format (without `python -m` prefix). The orchestrator template prepends `{{PYTHON_PATH}} -m` automatically.

## How It Works

1. For each issue, creates a branch `fix/issue-{id}-{operator}` in a worktree at `.worktrees/fix-{id}-{operator}`
2. Launches a Claude Code session with the `templates/fix_issue.md` prompt
3. CC reproduces the error, diagnoses root cause, implements fix, and validates
4. Results are parsed from CC's stream-json output and written to `results/summary.json`
5. On failure, retries up to `max_retries` total attempts (including the initial attempt)
6. If CC output is not parseable but the worktree has changes, the issue is marked `needs_review` for manual inspection

## Output

```
results/
├── summary.json                                    # Overall run summary
└── logs/
    ├── issue-418-sparse_mla_fwd_interface.jsonl    # Raw CC stream-json output
    ├── issue-418-sparse_mla_fwd_interface.log      # CC stderr
    └── issue-418-sparse_mla_fwd_interface.timeline.txt  # Human-readable timeline
```

### summary.json schema

```json
{
  "start_time": "2024-01-01T00:00:00+00:00",
  "end_time": "2024-01-01T01:00:00+00:00",
  "summary": {
    "total": 5,
    "success": 3,
    "failed": 1,
    "needs_review": 1,
    "in_progress": 0
  },
  "issues": {
    "issue-418": {
      "issue_id": 418,
      "operator": "sparse_mla_fwd_interface",
      "status": "success",
      "gpu_id": 0,
      "attempt": 1,
      "worktree_path": "...",
      "branch": "fix/issue-418-sparse_mla_fwd_interface",
      "duration_seconds": 600,
      "test_passed": true,
      "benchmark_passed": true,
      "format_check_passed": true,
      "cc_result": { "..." : "..." }
    }
  }
}
```

## Configuration

See `config.yaml.example` for all available options.

| Option | Default | Description |
|--------|---------|-------------|
| `flaggems_dir` | (auto-detected) | Path to FlagGems git repo |
| `python_path` | `python` | Python interpreter with FlagGems environment |
| `claude_bin` | `claude` | Claude Code executable |
| `budget_per_op` | 10000000.0 | Max budget (USD) per issue per attempt |
| `max_retries` | 2 | Total attempts per issue (including initial) |
| `timeout_per_op` | 3600 | Seconds before killing a stuck session |
| `poll_interval` | 10 | Seconds between process status checks |
| `base_branch` | master | Branch to create worktrees from |
| `gems_vendor` | nvidia | FlagGems vendor override for the environment |
| `template` | templates/fix_issue.md | Prompt template path (relative to script dir) |
| `results_dir` | results | Output directory (relative to script dir) |
| `device.lock_dir` | /tmp/auto_fix_gpu_locks | Lock file directory for GPU allocation |
| `device.gpu_ids` | null (auto-detect) | GPUs to use, e.g. `[0, 1, 2, 3]` |

## Server Mode (Distributed)

For multi-machine setups, the server mode provides a **coordinator-worker architecture** that distributes issues across machines by hardware type.

```
                    ┌──────────────┐
                    │  Coordinator │  :8000
                    │  (调度中心)   │
                    └──┬───────┬───┘
                       │       │
            ┌──────────┘       └──────────┐
            ▼                             ▼
    ┌───────────────┐             ┌───────────────┐
    │  Worker (H800) │  :8100     │  Worker (A100) │  :8200
    │  GPU 管理       │            │  GPU 管理       │
    │  Issue 执行     │            │  Issue 执行     │
    └───────────────┘             └───────────────┘
```

### Quick Start (Server Mode)

**1. Configure workers**

Each worker machine needs a worker config:

```bash
cp server/worker_config.yaml.example server/worker_config.yaml
# Edit: set hardware_type, port, and orchestrator_config path
```

**2. Configure coordinator**

The coordinator machine registers all workers:

```bash
cp server/coordinator_config.yaml.example server/coordinator_config.yaml
# Edit: add worker entries with id, url, and hardware_type
```

**3. Start workers** (on each GPU machine)

```bash
python -m server worker -c server/worker_config.yaml [-v]
```

**4. Start coordinator**

```bash
python -m server coordinator -c server/coordinator_config.yaml [-v]
```

**5. Submit issues**

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "issues": [
      {
        "id": "418",
        "operator": "sparse_mla_fwd_interface",
        "type": "accuracy_fail",
        "test_cmd": "pytest -m sparse_mla_fwd_interface tests/ --ref cpu -vs",
        "hardware": "H800"
      }
    ]
  }'
```

### API Reference

#### Coordinator Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST /jobs` | Submit issues for fixing | Routes to workers by `hardware` field |
| `GET /jobs` | List all jobs | Filter by `?status=`, `?hardware=`, `?worker_id=` |
| `GET /jobs/{job_id}` | Get job detail | Fetches latest status from worker |
| `DELETE /jobs/{job_id}` | Cancel a job | Cascades cancel to worker |
| `GET /workers` | List registered workers | Shows health, GPU availability |

#### Worker Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET /status` | Worker status | GPU count, available GPUs, queue depth |
| `POST /jobs` | Submit issues directly | Bypasses coordinator |
| `GET /jobs` | List jobs on this worker | Filter by `?status=` |
| `GET /jobs/{job_id}` | Get job detail | |
| `DELETE /jobs/{job_id}` | Cancel a job | |

All endpoints support optional `X-API-Key` header authentication (configure `api_key` in config).

### Server Mode Configuration

**Worker config** (`server/worker_config.yaml`):

| Option | Default | Description |
|--------|---------|-------------|
| `hardware_type` | (required) | GPU type identifier (e.g. `H800`, `A100`) |
| `host` | `0.0.0.0` | Bind address |
| `port` | `8100` | Listen port |
| `api_key` | `null` | Optional API key for endpoint auth |
| `orchestrator_config` | `../config.yaml` | Path to the orchestrator config (relative to this file) |

**Coordinator config** (`server/coordinator_config.yaml`):

| Option | Default | Description |
|--------|---------|-------------|
| `host` | `0.0.0.0` | Bind address |
| `port` | `8000` | Listen port |
| `api_key` | `null` | Optional API key for endpoint auth |
| `workers` | (required) | List of `{id, url, hardware_type}` entries |
| `worker_api_keys` | `{}` | Per-worker API keys (`worker_id → key`) |
| `poll_interval` | `30` | Health check interval (seconds) |
| `job_poll_interval` | `15` | Job status sync interval (seconds) |
| `unhealthy_threshold` | `3` | Mark worker unhealthy after N consecutive failures |

### How Server Mode Works

1. **Coordinator** receives issue submissions via `POST /jobs`
2. Issues are **routed by `hardware` field** — round-robin across healthy workers of that type, preferring workers with more available GPUs
3. **Workers** queue issues locally, acquire GPUs from the device manager, and run the same fix pipeline as standalone mode (worktree → Claude Code → test → push)
4. Workers **auto-retry** failed jobs up to `max_retries` attempts
5. Coordinator **polls worker status** periodically to track job progress and detect unhealthy workers
6. Worker state is **persisted to disk** (`results/worker_state.json`) — jobs in progress during a restart are marked as failed

### Job Status Flow

```
queued → running → success
                 → failed (→ queued for retry)
                 → needs_review
                 → cancelled
```

## Dependencies

- Python 3.10+
- `pyyaml` (`pip install pyyaml`)
- Claude Code CLI (`claude`) installed and on PATH
- GPU environment with `nvidia-smi` available
- **Server mode only**: `fastapi`, `uvicorn`, `httpx`, `pydantic` (`pip install fastapi uvicorn httpx pydantic`)
