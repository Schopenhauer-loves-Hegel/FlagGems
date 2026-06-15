"""Adapter: runs a single issue through the orchestrator pipeline.

Imports functions from orchestrator.py without modification.
Designed to run synchronously in a thread pool executor.
"""

import logging
import os
import subprocess
import sys
import threading
import time

_parent_dir = os.path.join(os.path.dirname(__file__), "..")
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from orchestrator import (
    _kill_cc_process,
    create_worktree,
    generate_timeline,
    launch_cc,
    load_config,
    load_dotenv,
    parse_cc_result,
    post_commit_format_check,
    render_template,
)

logger = logging.getLogger(__name__)


def run_single_issue(
    issue_dict: dict,
    gpu_id: int,
    config: dict,
    template_path: str,
    log_dir: str,
    cancel_event: threading.Event,
    attempt: int = 0,
) -> dict:
    """Run the full fix pipeline for one issue. Returns a result dict.

    This is a blocking function meant to run in a thread pool.
    ``cancel_event`` is checked periodically to support cancellation.
    """
    issue_id = issue_dict["id"]
    operator = issue_dict.get("operator") or f"repo-{issue_id}"
    issue_dict.setdefault("operator", operator)
    task_name = f"issue-{issue_id}-{operator}"
    flaggems_dir = config.get(
        "flaggems_dir",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")),
    )
    base_branch = config.get("base_branch", "master")
    timeout = config.get("timeout_per_op", 3600)
    poll_interval = config.get("poll_interval", 10)
    python_path = config.get("python_path", "python")

    result = {
        "issue_id": str(issue_id),
        "operator": operator,
        "status": "failed",
        "gpu_id": gpu_id,
        "attempt": attempt,
        "branch": None,
        "worktree_path": None,
        "test_passed": None,
        "benchmark_passed": None,
        "format_check_passed": None,
        "error_message": None,
        "cc_result": None,
    }

    worktree_path = None
    branch_name = None
    proc = None

    try:
        worktree_path, branch_name = create_worktree(
            flaggems_dir, issue_dict, base_branch
        )
        result["worktree_path"] = worktree_path
        result["branch"] = branch_name

        proc = launch_cc(
            issue_dict,
            worktree_path,
            gpu_id,
            config,
            template_path,
            log_dir,
            attempt=attempt,
        )
        logger.info(f"[{task_name}] CC launched (PID={proc.pid}, GPU={gpu_id})")

        start_time = time.monotonic()
        while proc.poll() is None:
            if cancel_event.is_set():
                logger.info(f"[{task_name}] Cancellation requested")
                _kill_cc_process(proc)
                result["status"] = "cancelled"
                result["error_message"] = "Cancelled by user"
                return result

            elapsed = time.monotonic() - start_time
            if timeout and elapsed > timeout:
                logger.warning(f"[{task_name}] Timeout after {int(elapsed)}s")
                _kill_cc_process(proc)
                result["status"] = "failed"
                result["error_message"] = f"Timeout after {int(elapsed)}s"
                return result

            time.sleep(min(poll_interval, 5))

        exit_code = proc.returncode
        logger.info(f"[{task_name}] CC exited with code {exit_code}")

        cc_result = parse_cc_result(proc, issue_dict, worktree_path)
        result["cc_result"] = cc_result

        cc_status = cc_result.get("status", "failed")
        test_passed = cc_result.get("test_passed")
        bench_passed = cc_result.get("benchmark_passed")
        result["test_passed"] = test_passed
        result["benchmark_passed"] = bench_passed

        already_fixed = "already_fixed" in (cc_result.get("notes") or "")
        if already_fixed:
            result["status"] = "success"
            result["error_message"] = "Already fixed on base branch"
            return result

        if exit_code == 0 and cc_status not in ("failed",):
            fmt_log_path = os.path.join(
                log_dir,
                f"{task_name}.attempt-{attempt + 1}.format-check.log",
            )
            fmt_result = post_commit_format_check(
                worktree_path, python_path, log_path=fmt_log_path
            )
            if fmt_result:
                result["format_check_passed"] = fmt_result["passed"]
            else:
                result["format_check_passed"] = None

            timeline_jsonl = os.path.join(
                log_dir,
                f"{task_name}.attempt-{attempt + 1}.jsonl",
            )
            generate_timeline(timeline_jsonl, task_name)

            if test_passed:
                push_result = subprocess.run(
                    ["git", "push", "-u", "origin", branch_name, "--force"],
                    cwd=worktree_path,
                    capture_output=True,
                    text=True,
                )
                if push_result.returncode == 0:
                    result["status"] = "success"
                    logger.info(f"[{task_name}] Pushed branch {branch_name}")
                    origin_url = subprocess.run(
                        ["git", "remote", "get-url", "origin"],
                        cwd=worktree_path,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                    repo_url = origin_url.replace(".git", "")
                    result["branch_url"] = f"{repo_url}/tree/{branch_name}"
                else:
                    result["status"] = "needs_review"
                    logger.error(
                        f"[{task_name}] Push failed: {push_result.stderr}"
                    )
                    result["error_message"] = f"Push failed: {push_result.stderr.strip()}"
            elif cc_status == "needs_review":
                result["status"] = "needs_review"
            else:
                result["status"] = "failed"
                result["error_message"] = cc_result.get(
                    "error_message", "Tests did not pass"
                )
        else:
            result["status"] = "failed"
            result["error_message"] = cc_result.get(
                "error_message", f"CC exited with code {exit_code}"
            )

    except Exception as e:
        logger.exception(f"[{task_name}] Unhandled error")
        result["status"] = "failed"
        result["error_message"] = str(e)
    finally:
        if proc is not None:
            if proc.poll() is None:
                _kill_cc_process(proc)
            else:
                if hasattr(proc, "_stdout_file") and not proc._stdout_file.closed:
                    proc._stdout_file.close()
                if hasattr(proc, "_stderr_file") and not proc._stderr_file.closed:
                    proc._stderr_file.close()

    return result
