"""Research agent: answers a question by searching and reading the web.

Uses Anthropic's server-side web_search and web_fetch tools, so there is no
client-side tool loop to write - Claude issues searches and fetches on
Anthropic's infrastructure and folds the results into its answer.

Usage:
    python -m agents.research_agent "What changed in the latest Python release?"
"""

import sys

from agents.common import MODEL, get_client

TOOLS = [
    {"type": "web_search_20260209", "name": "web_search"},
    {"type": "web_fetch_20260209", "name": "web_fetch"},
]


def research(question: str) -> str:
    client = get_client()
    messages = [{"role": "user", "content": question}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        tools=TOOLS,
        messages=messages,
    )

    # A long research turn can hit the server-side tool-call iteration limit
    # and come back with stop_reason "pause_turn" - resend to resume it.
    while response.stop_reason == "pause_turn":
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response.content},
        ]
        response = client.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=TOOLS,
            messages=messages,
        )

    sources = []
    answer_parts = []
    for block in response.content:
        if block.type == "text":
            answer_parts.append(block.text)
        elif block.type == "web_search_tool_result" and isinstance(block.content, list):
            for result in block.content:
                if getattr(result, "url", None):
                    sources.append((result.title, result.url))

    answer = "\n".join(answer_parts)
    if sources:
        seen = set()
        answer += "\n\nSources:\n"
        for title, url in sources:
            if url in seen:
                continue
            seen.add(url)
            answer += f"- {title}: {url}\n"

    return answer


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is the Claude API?"
    print(research(question))
