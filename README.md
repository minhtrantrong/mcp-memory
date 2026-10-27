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

Each MCP tool drives its matching agent directly — `ingest_memory` →
IngestAgent, `query_memory` → QueryAgent, `consolidate_memories` →
ConsolidateAgent — with no orchestrator hop, so a small local model only has to
make one tool decision. Behaviour you can rely on:

- `ingest_memory` checks the store after the model runs. If the model did not
  call `store_memory`, the text is saved verbatim without entity/topic
  extraction and the reply says so, instead of silently storing nothing.
- `consolidate_memories` answers "Nothing to consolidate" without calling the
  model when fewer than two unconsolidated memories exist.
- `memory_stats` is a direct database read and never calls the model.
- `entities`, `topics`, and `connections` are normalised to lists, so a model
  that sends one comma-separated string still searches and renders correctly.


## Requirements

- Python 3.11+
- `uv`
- [LM Studio](https://lmstudio.ai) with a model loaded and its local server running
  (OpenAI-compatible API at `http://127.0.0.1:1234/v1`):

```powershell
lms server start
lms ls                      # list downloaded model keys
```

The project pins the MCP SDK to its 1.x API because the server uses `FastMCP`.
The agents run on Google ADK and use a local LM Studio model by default:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEMORY_PROVIDER` | `lmstudio` | Model backend (`lmstudio`, `ollama`, or `gemini`) |
| `MODEL` | `liquid/lfm2.5-1.2b` | Model key or name |
| `LMSTUDIO_API_BASE` | `http://127.0.0.1:1234/v1` | LM Studio server endpoint |
| `LMSTUDIO_API_KEY` | `lm-studio` | Placeholder key; LM Studio ignores the value |
| `OLLAMA_API_BASE` | `http://127.0.0.1:11434` | Ollama endpoint, used only when `MEMORY_PROVIDER=ollama` |

LM Studio is reached through ADK's `LiteLlm` adapter over its OpenAI-compatible
API (`openai/<MODEL>`), so no real API key is needed. `MODEL` must match a model
key that LM Studio has loaded:

```powershell
curl http://127.0.0.1:1234/v1/models
```

Small local models follow the tool-calling instructions less reliably than
hosted models; set `MODEL` to a larger tool-capable model if ingest or
consolidation results look incomplete.

`litellm` is pinned below 1.106 because >=1.106 imports an MCP gateway module
that requires `mcp` 2.x, which conflicts with the `mcp` 1.x pin the server needs
for `FastMCP`. `tests/test_model_provider.py` guards that combination.

To use Ollama instead:

```powershell
$env:MEMORY_PROVIDER = "ollama"
$env:MODEL = "qwen2.5:1.6b"   # any pulled Ollama tag
```

To use Gemini, switch the provider and provide Google AI API credentials or
configure Vertex AI:

```powershell
$env:MEMORY_PROVIDER = "gemini"
$env:GOOGLE_API_KEY = "your-api-key"
```

Media ingestion (images, audio, video, PDFs) requires `MEMORY_PROVIDER=gemini`;
with LM Studio or Ollama those files are skipped with a warning because the
configured local models only accept text.

## Run locally

From the repository root:

```powershell
uv sync
uv run python server.py
```

Make sure LM Studio's local server is running and the model in `MODEL` is loaded
before the first `ingest_memory`, `query_memory`, or `consolidate_memories` call.
If the server listens elsewhere, override the endpoint:

```powershell
$env:LMSTUDIO_API_BASE = "http://127.0.0.1:1234/v1"
```

Never put credentials in `.mcp.json` or commit them.

The service binds only to `127.0.0.1` and listens on port `8765` by default.
Set `MEMORY_MCP_PORT` to choose another local port. The server creates the
SQLite database on startup at `memory.db` next to [server.py](./server.py), and
the dashboard resolves that same file, so both processes agree even when
launched from another directory. Set `MEMORY_DB` to use a different path in both
processes.

On Windows, CPython's asyncio Proactor transport logs
`ConnectionResetError: [WinError 10054]` from
`_ProactorBasePipeTransport._call_connection_lost` when a browser tab or MCP
client drops a connection abruptly. The connection is already gone at that
point, so [asyncio_noise.py](./asyncio_noise.py) drops that single log record in
both entry points; the dashboard and server keep running normally.

Keep the process running before starting or reconnecting Claude Code. The
project-level `.mcp.json` points to `http://127.0.0.1:8765/mcp`.

## Monitor memory with the dashboard

The upstream dashboard calls the original agent's REST API; this version
instead reads the MCP server's SQLite database directly in read-only mode.
It displays memory statistics, recent memories, and consolidation history,
refreshing automatically every 15 seconds. It does not ingest, query the model,
consolidate, or modify/delete memory.

Start the MCP server as above, then open a second terminal from the repository
root and run:

```powershell
uv run python -m streamlit run dashboard.py --server.address 127.0.0.1 --server.port 8501
```

Open `http://127.0.0.1:8501`. The dashboard uses the same default database as
the MCP server. If `MEMORY_DB` is set, set it to the same path in both
processes. The dashboard can be started before the first memory database is
created; it will show a waiting message and refresh automatically.

### Switch provider and model from the dashboard

The two dropdowns at the top of the page choose the API provider
(`lmstudio`, `ollama`, `gemini`) and the model. Model ids are read live from the
running local server, and you can type a name that is not listed. The choice is
saved to `memory-config.json` next to [server.py](./server.py) (override the
location with `MEMORY_CONFIG`), and the MCP server picks it up on its next tool
call — no restart needed. Precedence is: dashboard selection, then
`MEMORY_PROVIDER`/`MODEL`, then the built-in defaults.

## Test

```powershell
uv run python -m unittest discover -s tests -v
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
