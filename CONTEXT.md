# CONTEXT.md — sdk (cambrian-agent-sdk, the Python SDK for Cambrian agents)

The Python package `cambrian-agent-sdk` (PyPI, own version line, Apache-2.0, Python >= 3.10). Authors subclass
`CognitiveAgent` or `DaemonAgent`, decorate `@tool` methods and call `agent.serve()`; the kernel discovers the
agent file and runs it as its own process. The vendored gRPC proto in `cambrian_agent_sdk/_proto/` is the only
contract with the Go kernel: the SDK never imports the kernel.

## Build and test

```sh
uv sync --extra dev   # or: pip install -e ".[dev]"
pytest                # tests
ruff check            # lint ([tool.ruff] in pyproject.toml; _proto/ excluded)
mypy                  # types ([tool.mypy]; _proto/ excluded)
```

No CI workflow runs these; run them yourself before calling a change clean.

## Where the documentation is
- **Areas of this repo** are documented by knowledge cards in HQ. For any file or folder run
  `brief <path>` (HQ MCP tool, or `C:/Users/afsin/Dev/hq/bin/hq.exe brief <path> --config hq.yaml` from the
  workspace root); a card also loads automatically when you open a matching file. Cards are the area documentation.
- **Changing what a card says**: `read_doc cards/<name>.md`, edit, `write_doc` with a reason. Do not add facts here.
- **Why it is this shape**: ADRs and decisions in HQ (`search_docs`, `read_doc adrs/…`; status `list_decisions`).
- **Open work and known gaps**: HQ tickets (`list_tickets repo=sdk`).
- **Rules for agents**: [`AGENTS.md`](AGENTS.md) and the workspace [`../AGENTS.md`](../AGENTS.md).
