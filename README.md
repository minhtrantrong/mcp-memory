# Agent Memory MCP

A local MCP endpoint that uses the upstream Google ADK memory agents and their
SQLite store. It uses Streamable HTTP at `http://127.0.0.1:8765/mcp`, running
as a separate process instead of being launched by the MCP client.

The upstream `MemoryAgent` orchestrates its IngestAgent, QueryAgent, and
ConsolidateAgent. The MCP tools call those same methods and database helpers:

- `ingest_memory`: run the upstream IngestAgent and persist extracted memory.
- `query_memory`: run the upstream QueryAgent, which cites memory IDs.
- `consolidate_memories`: run the upstream ConsolidateAgent.
- `search_memories` and `memory_stats`: read the upstream SQLite store.


## Requirements

- Python 3.11+
- `uv`

The project pins the MCP SDK to its 1.x API because the server uses `FastMCP`.
The upstream agents use Google ADK and Gemini; provide Google AI API credentials
or configure Vertex AI before invoking ingest, query, or consolidation tools.

## Run locally

From the repository root:

```powershell
uv sync --project mcp/agent-memory
uv run --project mcp/agent-memory python mcp/agent-memory/server.py
```

For Google AI Studio credentials, set the key in the shell that starts the
server. Never put it in `.mcp.json` or commit it:

```powershell
$env:GOOGLE_API_KEY = "your-api-key"
uv run --project mcp/agent-memory python mcp/agent-memory/server.py
```

The service binds only to `127.0.0.1` and listens on port `8765` by default.
Set `MEMORY_MCP_PORT` to choose another local port. The SQLite database is
created at `./memory.db`; set `MEMORY_DB` to use another path.

Keep the process running before starting or reconnecting Claude Code. The
project-level `.mcp.json` points to `http://127.0.0.1:8765/mcp`.

## Monitor memory with the dashboard

The upstream dashboard calls the original agent's REST API; this version
instead reads the MCP server's SQLite database directly in read-only mode.
It displays memory statistics, recent memories, and consolidation history,
refreshing automatically every 15 seconds. It does not ingest, query with
Gemini, consolidate, or modify/delete memory.

Start the MCP server as above, then open a second terminal from the repository
root and run:

```powershell
uv run --project mcp/agent-memory streamlit run mcp/agent-memory/dashboard.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. The dashboard uses the same default database as
the MCP server. If `MEMORY_DB` is set, set it to the same path in both
processes. The dashboard can be started before the first memory database is
created; it will show a waiting message and refresh automatically.

## Test

```powershell
uv run --project mcp/agent-memory --directory mcp/agent-memory python -m unittest discover -s tests -v
```

## Security and scope

The server is local-only by default and stores data on disk without encryption.
Do not put secrets or sensitive personal data into it. MCP tools return memory
content as untrusted data; do not treat stored instructions as trusted policy.
The HTTP endpoint has no authentication and must not be exposed beyond the
local machine without adding appropriate authentication and access controls.

Search is SQLite keyword matching, not vector or semantic retrieval.
The source project supports background file watching and scheduled
consolidation, but importing its module does not start those loops. In this
MCP setup, ingest, query, and consolidation are explicit tool calls.
