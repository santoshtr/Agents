"""Shared setup for the example agents."""

import anthropic

MODEL = "claude-opus-5"


def get_client() -> anthropic.Anthropic:
    """Anthropic() resolves credentials from ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / an `ant auth login` profile."""
    return anthropic.Anthropic()
