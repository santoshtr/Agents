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



import csv
import sys
from datetime import datetime
from pathlib import Path

from agents.common import MODEL, PRICES, get_client

LOG_FILE = Path(__file__).resolve().parent.parent / "usage_log.csv"

TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 3},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 5},
]


def _dedup(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    seen = set()
    out = []
    for title, url in pairs:
        if url not in seen:
            seen.add(url)
            out.append((title, url))
    return out


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
        cited_sources = []      # URLs Claude explicitly attributed a sentence to
        searched_sources = []   # every page a search turned up, cited or not

        for block in response.content:
            if block.type == "text":
                answer_parts.append(block.text)
                for citation in block.citations or []:
                    if getattr(citation, "url", None):
                        cited_sources.append((citation.title or citation.url, citation.url))
            elif block.type == "web_search_tool_result" and isinstance(block.content, list):
                for result in block.content:
                    url = getattr(result, "url", None)
                    if url:
                        searched_sources.append((getattr(result, "title", None) or url, url))

        answer = "\n".join(answer_parts)
        cited_sources = _dedup(cited_sources)

        if cited_sources:
            # Best case: Claude told us exactly which sentence came from where.
            answer += "\n\nSources:\n"
            for title, url in cited_sources:
                answer += f"- {title}: {url}\n"
        else:
            # Claude didn't attach citations this turn (it doesn't always).
            # Fall back to what it searched, but say plainly these are
            # unverified - not confirmed as the basis for any specific claim.
            searched_sources = _dedup(searched_sources)[:10]
            if searched_sources:
                answer += "\n\n(No sources were explicitly cited this run. Pages searched, unverified:)\n"
                for title, url in searched_sources:
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
