"""Shared setup for the example agents."""

import os
import sys
from pathlib import Path

import anthropic

MODEL = "claude-opus-5"

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path: Path = ENV_FILE) -> None:
    """Load KEY=value lines from a .env file into the environment.

    Real environment variables always win, so `export ANTHROPIC_API_KEY=...`
    still overrides the file. Kept dependency-free on purpose - this is the
    whole of what python-dotenv would do for us here.
    """
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def get_client() -> anthropic.Anthropic:
    """Build a client, loading .env first and failing with a readable message."""
    load_env()

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        sys.exit(
            "No API key found.\n\n"
            "Create a key at https://console.anthropic.com/settings/keys, then either:\n"
            f"  1. Copy .env.example to .env and paste the key in ({ENV_FILE.parent}/.env), or\n"
            "  2. Run: export ANTHROPIC_API_KEY=sk-ant-...\n"
        )

    return anthropic.Anthropic()
