# FlagGems Skills

Claude Code skills for FlagGems development workflow.

## Installation

### Manual Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Schopenhauer-loves-Hegel/FlagGems.git
   cd FlagGems
   git checkout skills
   ```

2. Copy skills to your Claude Code skills directory:
   ```bash
   cp -r skills/fetch-issues ~/.claude/skills/
   ```

3. Use the skill in Claude Code:
   ```
   /fetch-issues
   ```

### Plugin Installation (if supported)

```bash
claude plugin marketplace add https://github.com/Schopenhauer-loves-Hegel/FlagGems --branch skills
claude plugin install fetch-issues@flaggems-skills
```

## Available Skills

### fetch-issues

Query the FlagGems internal issue tracking system (http://10.1.4.213:31080).

**Features:**
- Fetch issues assigned to you
- View statistics (open/closed counts, trends)
- Filter by status, severity, assignee
- JSON or table output

**Quick usage:**
```bash
# Your open issues
bash skills/fetch-issues/fetch_issues.sh

# Statistics
bash skills/fetch-issues/fetch_issues.sh --stats

# All open issues
bash skills/fetch-issues/fetch_issues.sh --all
```

See [fetch-issues/SKILL.md](fetch-issues/SKILL.md) for full documentation.

## Configuration

### Environment Variables

| Var | Description | Default |
|-----|-------------|---------|
| `ISSUES_URL` | Issue tracking system URL | http://10.1.4.213:31080 |
| `ISSUES_USER` | Login username | admin |
| `ISSUES_PASS` | Login password | admin123 |

### Custom Assignee ID

To use a different assignee ID, modify the `ASSIGNED_TO` variable in `fetch-issues/fetch_issues.sh` or pass `--assigned-to`:

```bash
bash skills/fetch-issues/fetch_issues.sh --assigned-to 42
```

## Development

### Adding New Skills

1. Create a new directory under `skills/`:
   ```bash
   mkdir skills/my-new-skill
   ```

2. Add `SKILL.md` with frontmatter:
   ```markdown
   ---
   name: my-new-skill
   description: What this skill does and when to use it.
   ---

   # My New Skill

   Content here...
   ```

3. Add scripts if needed

4. Update `.claude-plugin/marketplace.json` to register the skill

5. Update this README

### Skill Structure

```
skills/
├── .claude-plugin/
│   └── marketplace.json    # Plugin registry
├── fetch-issues/
│   ├── SKILL.md            # Skill definition with frontmatter
│   └── fetch_issues.sh     # Implementation script
├── my-new-skill/
│   ├── SKILL.md
│   └── scripts/
└── README.md               # This file
```

## Links

- [FlagGems Repository](https://github.com/flagos-ai/FlagGems)
- [Issue Tracking System](http://10.1.4.213:31080)
- [Claude Code Skills Documentation](https://docs.anthropic.com/claude-code/skills)
