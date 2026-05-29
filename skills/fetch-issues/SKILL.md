---
name: fetch-issues
description: Query FlagGems internal issue tracking system (http://10.1.4.213:31080) to fetch issues assigned to you, view statistics, and filter by status/severity. Use when checking open issues, reviewing accuracy failures, or tracking issue progress.
---

# Fetch Issues

Query the FlagGems internal issue tracking system and display results.

## When to Use This Skill

This skill activates when:
- Checking open issues assigned to you
- Reviewing accuracy test failures
- Tracking issue statistics and trends
- Filtering issues by status, severity, or assignee

## Quick Start

```bash
# Show open issues assigned to you (default)
bash skills/fetch-issues/fetch_issues.sh

# Show statistics summary
bash skills/fetch-issues/fetch_issues.sh --stats

# Show all open issues (no assignee filter)
bash skills/fetch-issues/fetch_issues.sh --all
```

## Usage

```bash
bash skills/fetch-issues/fetch_issues.sh [OPTIONS]
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--assigned-to ID` | Filter by assignee user ID | 22 (Schopenhauer-loves-Hegel) |
| `--status STATUS` | open, in_progress, resolved, closed | open |
| `--page-size N` | Results per page | 100 |
| `--stats` | Show statistics summary | - |
| `--all` | No assignee filter | - |
| `--format` | table or json | table |
| `--help` | Show help | - |

### Environment Variables

| Var | Description | Default |
|-----|-------------|---------|
| `ISSUES_URL` | Base URL | http://10.1.4.213:31080 |
| `ISSUES_USER` | Login username | admin |
| `ISSUES_PASS` | Login password | admin123 |

## Examples

### Check your open issues

```bash
bash skills/fetch-issues/fetch_issues.sh
```

Output:
```
=== Issues (8 total) ===
   ID | Operator                            | Type             | Severity | Backend  |   Age | Status
----------------------------------------------------------------------------------------------------
  418 | sparse_mla_fwd_interface            | accuracy_fail    | major    | H800     |    1d | open
  417 | leaky_relu_                         | accuracy_fail    | major    | H800     |    1d | open
...
```

### View statistics

```bash
bash skills/fetch-issues/fetch_issues.sh --stats
```

Output:
```
=== Issue Statistics ===
Open: 230  |  Closed: 188
New this week: 18
Avg age: 105.8 days
Overdue (>14d): 212

By assignee:
  Schopenhauer-loves-Hegel: open=8 in_progress=0 resolved=0
  未指派: open=222 in_progress=0 resolved=0
```

### JSON output for programmatic use

```bash
bash skills/fetch-issues/fetch_issues.sh --format json | jq '.items[] | {id, title, status}'
```

## API Reference

The script uses the following API endpoints:

- `POST /api/auth/login` - Authenticate and get JWT token
- `GET /api/issues/` - List issues with filters
- `GET /api/issues/stats` - Get statistics summary

## Related

- Issue tracking system: http://10.1.4.213:31080/issues
- FlagGems repository: https://github.com/flagos-ai/FlagGems
