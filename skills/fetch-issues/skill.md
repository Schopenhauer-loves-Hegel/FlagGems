---
name: fetch-issues
description: Fetch issues from FlagGems internal issue tracking system (http://10.1.4.213:31080)
---

# fetch-issues

Query the FlagGems internal issue tracking system and display results.

## Usage

Run the script at `skills/fetch-issues/fetch_issues.sh` with optional filters:

```bash
# Show open issues assigned to current user (default)
bash skills/fetch-issues/fetch_issues.sh

# Show statistics
bash skills/fetch-issues/fetch_issues.sh --stats

# Show all open issues (not just assigned)
bash skills/fetch-issues/fetch_issues.sh --all

# Filter by status
bash skills/fetch-issues/fetch_issues.sh --status in_progress

# JSON output
bash skills/fetch-issues/fetch_issues.sh --format json
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `--assigned-to ID` | Filter by assignee user ID | 22 (Schopenhauer-loves-Hegel) |
| `--status STATUS` | open, in_progress, resolved, closed | open |
| `--page-size N` | Results per page | 100 |
| `--stats` | Show statistics summary | - |
| `--all` | No assignee filter | - |
| `--format` | table or json | table |

## Environment Variables

| Var | Description | Default |
|-----|-------------|---------|
| `ISSUES_URL` | Base URL | http://10.1.4.213:31080 |
| `ISSUES_USER` | Login username | admin |
| `ISSUES_PASS` | Login password | admin123 |
