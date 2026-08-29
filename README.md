# Agents

Agents that run tasks, built on the Claude API.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # or `ant auth login`
```

## Examples

Three runnable agents, each demonstrating a different pattern for putting an
LLM in a loop:

### 1. `agents/research_agent.py` — server-side tools

Answers a question by searching and reading the web. Uses Anthropic's
server-side `web_search` and `web_fetch` tools, so there's no client-side
tool loop to write — Claude issues searches and fetches on Anthropic's
infrastructure and returns an answer with sources.

```bash
python -m agents.research_agent "What changed in the latest Python release?"
```

### 2. `agents/task_agent.py` — custom tools via the Tool Runner

A general task agent with two of its own tools (a calculator and a
note-saver). Tools are plain Python functions decorated with `@beta_tool`;
the SDK's Tool Runner drives the call/execute/feed-results-back loop.
This is the pattern to copy for "agent with my own tools" use cases.

```bash
python -m agents.task_agent "Split \$137.50 three ways and save the result as a note called split"
```

### 3. `agents/file_editor_agent.py` — a sandboxed coding agent

Uses Anthropic's built-in `bash` and text-editor tools to read, write, and
edit files to complete a coding task. These tools are schema-less and
client-executed, so this example hand-writes the agentic loop and confines
every file operation and shell command to `./workspace` with a small
command allowlist.

```bash
python -m agents.file_editor_agent "Create hello.py that prints Hello, agents!"
```

## Where to go from here

Once one of these fits, some natural next steps:

- **Managed Agents** — let Anthropic host the agent loop *and* the
  execution sandbox (per-session containers, scheduled runs, persisted
  agent configs) instead of running the loop yourself.
- **MCP (Model Context Protocol)** — connect an agent to existing tool
  servers instead of hand-writing every tool.
- **Claude Agent SDK** — a batteries-included coding/filesystem agent (the
  Claude Code harness as a library) when you want built-in Read/Write/Bash/
  Grep tools rather than defining your own.
