"""Research agent: answers questions by searching and reading the web.

Uses Anthropic's server-side web_search and web_fetch tools, so there is no
client-side tool loop to write - Claude issues searches and fetches on
Anthropic's infrastructure and folds the results into its answer.

Usage:
    # One question, then exit:
    python -m agents.research_agent "What changed in the latest Python release?"

    # Interactive - ask follow-ups that remember the conversation:
    python -m agents.research_agent
"""



import sys
from urllib import response
from agents.common import MODEL, PRICES, get_client


import csv
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "usage_log.csv"

TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 3},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 5},
]

from urllib.parse import urlparse


class ResearchAgent:
    """Holds the conversation so follow-up questions keep their context.

    The API itself is stateless - "memory" is just us resending the whole
    message history on each turn, which is what self.messages accumulates.
    """

    def __init__(self):
        self.client = get_client()
        self.messages = []

    def ask(self, question: str) -> str:
        self.messages.append({"role": "user", "content": question})

        response = self.client.messages.create(
            model=MODEL,
            max_tokens=16000,
            tools=TOOLS,
            messages=self.messages,
        )

        # A long research turn can hit the server-side tool-call iteration
        # limit and come back with stop_reason "pause_turn". Appending the
        # partial turn and resending lets the server pick up where it left off.
        pauses = 0
        while response.stop_reason == "pause_turn" and pauses < 5:
            self.messages.append({"role": "assistant", "content": response.content})
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=16000,
                tools=TOOLS,
                messages=self.messages,
            )
            pauses += 1

        # Keep the full content (not just the text) so search results stay in
        # context for follow-up questions.
        self.messages.append({"role": "assistant", "content": response.content})
        in_price, out_price = PRICES[MODEL]
        cost = response.usage.input_tokens * in_price / 1_000_000 + response.usage.output_tokens * out_price / 1_000_000
        is_new = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "model", "input_tokens", "output_tokens", "cost_usd", "question"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), response.model,
                            response.usage.input_tokens, response.usage.output_tokens,
                            f"{cost:.4f}", question[:80]])
    
        print(f"[usage: {response.model} · {response.usage.input_tokens} in / {response.usage.output_tokens} out · ${cost:.2f}]")
        return self._format(response)

    @staticmethod
    def _format(response) -> str:
        answer_parts = []
        sources = []

        for block in response.content:
            if block.type != "text":
                continue
            answer_parts.append(block.text)
            for citation in (block.citations or []):
                if getattr(citation, "url", None):
                    sources.append((citation.title, citation.url))

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

def interactive() -> None:
    agent = ResearchAgent()
    print("Research agent. Ask a question, or press Ctrl-C / type 'exit' to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            return

        print("\nsearching...\n")
        print(agent.ask(question))
        print()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:])
    if question:
        print(ResearchAgent().ask(question))
    else:
        interactive()
