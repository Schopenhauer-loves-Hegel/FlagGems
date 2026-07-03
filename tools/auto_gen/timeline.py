"""Timeline generation from CC stream-json logs."""

import json
import logging

logger = logging.getLogger(__name__)


def generate_timeline(jsonl_path: str, operator: str) -> str | None:
    """Generate a human-readable timeline from a CC stream-json log.

    Writes a .timeline.txt file next to the .jsonl and returns its path.
    """
    timeline_path = jsonl_path.replace(".jsonl", ".timeline.txt")
    try:
        events = []
        with open(jsonl_path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        out: list[str] = []
        step = 0

        def _format_tool_use(name: str, inp: dict) -> str:
            if name == "Bash":
                return inp.get("command", "")
            elif name in ("Read", "Write"):
                return inp.get("file_path", "")
            elif name == "Edit":
                s = inp.get("file_path", "")
                old = inp.get("old_string", "")
                new = inp.get("new_string", "")
                return f"{s}\n--- old ---\n{old}\n+++ new +++\n{new}"
            elif name in ("Grep", "Glob"):
                return f"pattern={inp.get('pattern', '')}  path={inp.get('path', '')}"
            else:
                return json.dumps(inp, ensure_ascii=False)

        for event in events:
            etype = event.get("type", "")

            if etype == "system" and event.get("subtype") == "init":
                out.append(f"=== {operator} ===")
                out.append(f"Session: {event.get('session_id', '?')}")
                out.append(f"Model: {event.get('model', '?')}")
                out.append("")
                continue

            if etype == "result":
                step += 1
                out.append(f"[{step}] ✅ Result:")
                out.append(event.get("result", ""))
                out.append("")
                continue

            if etype == "user":
                # Extract tool result output
                tool_result = event.get("tool_use_result")
                if isinstance(tool_result, dict):
                    output = tool_result.get("stdout", "") or tool_result.get(
                        "stderr", ""
                    )
                    if output:
                        out.append("    ↳ Output:")
                        out.append(str(output))
                        out.append("")
                        continue
                # Fallback: check message.content for tool_result entries
                contents = event.get("message", {}).get("content", [])
                if isinstance(contents, list):
                    for c in contents:
                        if isinstance(c, dict) and c.get("type") == "tool_result":
                            content_val = c.get("content", "")
                            if content_val:
                                out.append("    ↳ Output:")
                                out.append(str(content_val))
                                out.append("")
                            break
                continue

            if etype != "assistant":
                continue

            contents = event.get("message", {}).get("content", [])
            if not isinstance(contents, list):
                continue

            for content in contents:
                if not isinstance(content, dict):
                    continue
                ctype = content.get("type", "")

                if ctype == "thinking":
                    step += 1
                    out.append(f"[{step}] 🤔 Thinking:")
                    out.append(content.get("thinking", ""))
                    out.append("")

                elif ctype == "text":
                    text = content.get("text", "")
                    if text.strip():
                        step += 1
                        out.append(f"[{step}] 💬 Text:")
                        out.append(text)
                        out.append("")

                elif ctype == "tool_use":
                    step += 1
                    name = content.get("name", "?")
                    inp = content.get("input", {})
                    out.append(f"[{step}] 🔧 {name}:")
                    out.append(_format_tool_use(name, inp))
                    out.append("")

        with open(timeline_path, "w") as f:
            f.write("\n".join(out))

        logger.info(f"Generated timeline for {operator}: {timeline_path}")
        return timeline_path

    except Exception as e:
        logger.warning(f"Failed to generate timeline for {operator}: {e}")
        return None
