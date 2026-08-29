"""File-editing agent: a coding agent built on Anthropic's bash and text-editor tools.

Demonstrates the manual agentic loop needed for client-side, Anthropic-defined
tools (bash_20250124, text_editor_20250728) - these are schema-less and
executed locally, so the Tool Runner's decorator pattern doesn't apply; your
code has to dispatch and run them itself.

Everything the agent touches is confined to ./workspace, and bash is
restricted to a small allowlist of read-only/inspection commands. This is a
minimal safety net for a demo, not a production sandbox - see the Security
notes below before pointing this at anything untrusted.

Usage:
    python -m agents.file_editor_agent "Create hello.py that prints Hello, agents!"
"""

import shlex
import subprocess
import sys
from pathlib import Path

from agents.common import MODEL, get_client

ROOT = Path(__file__).resolve().parent.parent / "workspace"

# Security: bash is a client-executed tool driven by untrusted model output.
# Allow only a small set of inspection/run commands and reject shell
# operators outright - see shared/tool-use-concepts.md § Bash and Text Editor.
ALLOWED_COMMANDS = {"ls", "cat", "python3", "pytest", "echo", "mkdir", "pwd"}
SHELL_OPERATORS = ("&&", "||", "|", ";", "`", "$(", ">", "<")

TOOLS = [
    {"type": "bash_20250124", "name": "bash"},
    {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
]


def _resolve(path_str: str) -> Path:
    """Resolve a model-supplied path and confirm it stays inside ROOT."""
    candidate = (ROOT / path_str).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError(f"path '{path_str}' escapes the workspace root")
    return candidate


def handle_bash(tool_input: dict) -> str:
    if tool_input.get("restart"):
        return "Session restarted."

    command = tool_input.get("command", "")
    if any(op in command for op in SHELL_OPERATORS):
        return "Error: shell operators are not permitted."

    try:
        parts = shlex.split(command)
    except ValueError as exc:
        return f"Error: could not parse command: {exc}"

    if not parts or parts[0] not in ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(ALLOWED_COMMANDS))
        return f"Error: command not allowed. Allowed commands: {allowed}"

    try:
        result = subprocess.run(
            parts,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return "Error: command timed out."

    return (result.stdout + result.stderr) or "(no output)"


def handle_text_editor(tool_input: dict) -> str:
    command = tool_input.get("command")
    try:
        path = _resolve(tool_input["path"])
    except (KeyError, ValueError) as exc:
        return f"Error: {exc}"

    if command == "view":
        if path.is_dir():
            return "\n".join(sorted(p.name for p in path.iterdir()))
        if not path.exists():
            return f"Error: {path.relative_to(ROOT)} does not exist."
        return path.read_text()

    if command == "create":
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.with_suffix(path.suffix + ".bak").write_text(path.read_text())
        path.write_text(tool_input.get("file_text", ""))
        return f"Created {path.relative_to(ROOT)}"

    if command == "str_replace":
        text = path.read_text()
        old, new = tool_input["old_str"], tool_input["new_str"]
        count = text.count(old)
        if count != 1:
            return f"Error: expected exactly 1 match for old_str, found {count}."
        path.write_text(text.replace(old, new, 1))
        return f"Updated {path.relative_to(ROOT)}"

    if command == "insert":
        lines = path.read_text().splitlines(keepends=True)
        line_no = tool_input["insert_line"]
        insert_text = tool_input["insert_text"]
        if not insert_text.endswith("\n"):
            insert_text += "\n"
        lines.insert(line_no, insert_text)
        path.write_text("".join(lines))
        return f"Inserted into {path.relative_to(ROOT)}"

    return f"Error: unsupported command '{command}'."


def execute_tool(name: str, tool_input: dict) -> str:
    if name == "bash":
        return handle_bash(tool_input)
    if name == "str_replace_based_edit_tool":
        return handle_text_editor(tool_input)
    return f"Error: unknown tool '{name}'."


def run_task(instruction: str, max_turns: int = 20) -> str:
    ROOT.mkdir(exist_ok=True)
    client = get_client()
    messages = [{"role": "user", "content": instruction}]

    for _ in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return next((b.text for b in response.content if b.type == "text"), "")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                output = execute_tool(block.name, block.input)
                is_error = output.startswith("Error:")
            except Exception as exc:  # noqa: BLE001
                output, is_error = f"Error: {exc}", True
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    return "Stopped: reached max_turns without finishing."


if __name__ == "__main__":
    instruction = " ".join(sys.argv[1:]) or "Create hello.py that prints Hello, agents!"
    print(run_task(instruction))
