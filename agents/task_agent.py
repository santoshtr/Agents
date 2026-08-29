"""Task agent: a custom-tool agent built on the Tool Runner.

Gives Claude two small tools - a calculator and a note-saver - and lets it
decide when and how many times to call each while working through a
multi-step request. This is the pattern to copy for "agent with my own
tools" use cases: define plain Python functions, hand them to the tool
runner, and it drives the call/execute/feed-back loop for you.

Usage:
    python -m agents.task_agent "Split $137.50 three ways and save the result as a note called split"
"""

import re
import sys
from pathlib import Path

from anthropic import beta_tool

from agents.common import MODEL, get_client

NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"


@beta_tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: An arithmetic expression using digits and + - * / ( ) . only,
            e.g. "137.50 / 3".
    """
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expression):
        return "Error: expression contains unsupported characters."
    try:
        # Safe: the regex above only allows digits, whitespace, and arithmetic operators.
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


@beta_tool
def save_note(title: str, content: str) -> str:
    """Save a short note to disk so it can be found later.

    Args:
        title: A short filename-safe title for the note.
        content: The note's text content.
    """
    NOTES_DIR.mkdir(exist_ok=True)
    safe_title = re.sub(r"[^a-zA-Z0-9_-]+", "-", title).strip("-") or "note"
    path = NOTES_DIR / f"{safe_title}.txt"
    path.write_text(content)
    return f"Saved note to {path}"


def run_task(instruction: str) -> str:
    client = get_client()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=16000,
        tools=[calculate, save_note],
        messages=[{"role": "user", "content": instruction}],
    )

    final_text = ""
    for message in runner:
        for block in message.content:
            if block.type == "text":
                final_text = block.text
    return final_text


if __name__ == "__main__":
    instruction = " ".join(sys.argv[1:]) or "What is 47 * 12?"
    print(run_task(instruction))
