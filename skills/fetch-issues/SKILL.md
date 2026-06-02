---
name: fetch-issues
description: Query FlagGems internal issue tracking system (http://10.1.4.213:31080) to fetch issues assigned to you, view statistics, filter by status/severity, and retrieve test results for debugging accuracy failures.
---

# Fetch Issues

Query the FlagGems internal issue tracking system and display results.

## When to Use This Skill

This skill activates when:
- Checking open issues assigned to you
- Reviewing accuracy test failures
- Tracking issue statistics and trends
- Filtering issues by status, severity, or assignee
- Fetching test results for debugging (environment, accuracy, speedup)

## Scripts

### 1. fetch_issues.sh - Issue List

```bash
# Show open issues assigned to you (default)
bash skills/fetch-issues/fetch_issues.sh

# Show statistics summary
bash skills/fetch-issues/fetch_issues.sh --stats

# Show all open issues (no assignee filter)
bash skills/fetch-issues/fetch_issues.sh --all

# JSON output
bash skills/fetch-issues/fetch_issues.sh --format json
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--assigned-to ID` | Filter by assignee user ID | 22 (Schopenhauer-loves-Hegel) |
| `--status STATUS` | open, in_progress, resolved, closed | open |
| `--page-size N` | Results per page | 100 |
| `--stats` | Show statistics summary | - |
| `--all` | No assignee filter | - |
| `--format` | table or json | table |

### 2. fetch_test_results.sh - Test Results

```bash
# Fetch test context for a specific issue
bash skills/fetch-issues/fetch_test_results.sh --issue-id 418

# Fetch all results for a test run
bash skills/fetch-issues/fetch_test_results.sh --test-run-id 40

# Show only failed operators
bash skills/fetch-issues/fetch_test_results.sh --test-run-id 40 --failed-only

# JSON output
bash skills/fetch-issues/fetch_test_results.sh --issue-id 418 --format json
```

#### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--issue-id ID` | Fetch test context for a specific issue | - |
| `--test-run-id ID` | Fetch all results for a test run | - |
| `--failed-only` | Show only failed operators (with --test-run-id) | false |
| `--format` | table or json | table |

## Environment Variables

| Var | Description | Default |
|-----|-------------|---------|
| `ISSUES_URL` | Base URL | http://10.1.4.213:31080 |
| `ISSUES_USER` | Login username | admin |
| `ISSUES_PASS` | Login password | admin123 |

## API Endpoints

The scripts use these API endpoints:

- `POST /api/auth/login` - Authenticate and get JWT token
- `GET /api/issues/` - List issues with filters
- `GET /api/issues/stats` - Get statistics summary
- `GET /api/issues/{id}/test-context` - Get test environment and results for an issue
- `GET /api/reports/test-run/{id}/results` - Get all test results for a test run

## Debugging Workflow

When investigating accuracy failures:

1. **List your issues**: `bash fetch_issues.sh`
2. **Get test context**: `bash fetch_test_results.sh --issue-id <ID>`
   - Shows: accuracy pass/fail, test environment (PyTorch/Triton versions, GPU)
3. **Check test run**: `bash fetch_test_results.sh --test-run-id <ID> --failed-only`
   - Shows: all failed operators in that test run
4. **SSH to target machine** and reproduce:
   ```bash
   # Clone repo
   git clone https://github.com/flagos-ai/FlagGems.git
   cd FlagGems

   # Run specific test
   pytest tests/test_<operator>.py -v
   ```

## Example Output

### Issue Test Context
```
=== Test Context for Issue #418 ===

--- Test Result ---
  Status: FAIL
  Accuracy: 85/166 passed, 81 failed
  Speedup: N/A

--- Test Environment ---
  PyTorch: 2.9.0+cu128
  Device: NVIDIA H800 x8
  Triton: 3.6.0
  Python: 3.12.3
  FlagGems: 5.0.2+gitc27cee94
  OS: ubuntu 24.04
```

### Test Run Results (failed only)
```
=== Test Run #40 Results (8 operators) ===

Operator                            | Status | Passed | Failed | Total |  Speedup
--------------------------------------------------------------------------------
act_quant_triton                    | FAIL   |      0 |    112 |   112 |        -
addmm_dtype                         | FAIL   |      0 |      3 |     3 |   1.0281
...
```

## Related

- Issue tracking system: http://10.1.4.213:31080/issues
- FlagGems repository: https://github.com/flagos-ai/FlagGems
