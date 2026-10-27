"""
Agent Memory Layer — Always-On ADK Agent

A lightweight, cost-effective background agent that continuously processes, consolidates, and serves memory. Runs by default on a local LM Studio model, or on Ollama or Gemini 3.1 Flash-Lite when `MEMORY_PROVIDER` is set.

Usage:
    python agent.py                          # watch ./inbox, serve on :8888
    python agent.py --watch ./docs --port 9000
    python agent.py --consolidate-every 15   # consolidate every 15 min

Query:
    curl "http://localhost:8888/query?q=what+do+you+know"
    curl -X POST http://localhost:8888/ingest -d '{"text": "some info"}'
"""

import argparse
import asyncio
import json
import logging
import mimetypes
import os
import shutil
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from aiohttp import web
from google.adk.agents import Agent
from google.adk.models.base_llm import BaseLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from memory import config
from memory.records import as_dict_list, as_int_list, as_text_list, parse_json_list, parse_text_list

# ─── Config ────────────────────────────────────────────────────

PROVIDER, MODEL = config.env_selection()
LMSTUDIO_API_BASE = os.getenv("LMSTUDIO_API_BASE", config.LMSTUDIO_API_BASE).rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "lm-studio")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", config.OLLAMA_API_BASE).rstrip("/")
DB_PATH = os.getenv("MEMORY_DB", "memory.db")

# Supported file types for multimodal ingestion
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".log", ".xml", ".yaml", ".yml"}
MEDIA_EXTENSIONS = {
    # Images
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    # Audio
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    # Video
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    # Documents
    ".pdf": "application/pdf",
}
ALL_SUPPORTED = TEXT_EXTENSIONS | set(MEDIA_EXTENSIONS.keys())

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%H:%M]",
)
log = logging.getLogger("memory-agent")

# ─── Database ──────────────────────────────────────────────────


def get_db() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL DEFAULT '',
            raw_text TEXT NOT NULL,
            summary TEXT NOT NULL,
            entities TEXT NOT NULL DEFAULT '[]',
            topics TEXT NOT NULL DEFAULT '[]',
            connections TEXT NOT NULL DEFAULT '[]',
            importance REAL NOT NULL DEFAULT 0.5,
            created_at TEXT NOT NULL,
            consolidated INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS consolidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_ids TEXT NOT NULL,
            summary TEXT NOT NULL,
            insight TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS processed_files (
            path TEXT PRIMARY KEY,
            processed_at TEXT NOT NULL
        );
    """)
    return db


# ─── ADK Tools ─────────────────────────────────────────────────


def store_memory(
    raw_text: str,
    summary: str,
    entities: list[str],
    topics: list[str],
    importance: float,
    source: str = "",
) -> dict:
    """Store a processed memory in the database.

    Args:
        raw_text: The original input text.
        summary: A concise 1-2 sentence summary.
        entities: Key people, companies, products, or concepts.
        topics: 2-4 topic tags.
        importance: Float 0.0 to 1.0 indicating importance.
        source: Where this memory came from (filename, URL, etc).

    Returns:
        dict with memory_id and confirmation.
    """
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    cursor = db.execute(
        """INSERT INTO memories (source, raw_text, summary, entities, topics, importance, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            source,
            raw_text,
            summary,
            json.dumps(as_text_list(entities)),
            json.dumps(as_text_list(topics)),
            importance,
            now,
        ),
    )
    db.commit()
    mid = cursor.lastrowid
    db.close()
    log.info(f"📥 Stored memory #{mid}: {summary[:60]}...")
    return {"memory_id": mid, "status": "stored", "summary": summary}


def read_all_memories() -> dict:
    """Read all stored memories from the database, most recent first.

    Returns:
        dict with list of memories and count.
    """
    db = get_db()
    rows = db.execute("SELECT * FROM memories ORDER BY created_at DESC LIMIT 50").fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "source": r["source"], "summary": r["summary"],
            "entities": parse_text_list(r["entities"]), "topics": parse_text_list(r["topics"]),
            "importance": r["importance"], "connections": parse_json_list(r["connections"]),
            "created_at": r["created_at"], "consolidated": bool(r["consolidated"]),
        })
    db.close()
    return {"memories": memories, "count": len(memories)}


def read_unconsolidated_memories() -> dict:
    """Read memories that haven't been consolidated yet.

    Returns:
        dict with list of unconsolidated memories and count.
    """
    db = get_db()
    rows = db.execute(
        "SELECT * FROM memories WHERE consolidated = 0 ORDER BY created_at DESC LIMIT 10"
    ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "summary": r["summary"],
            "entities": parse_text_list(r["entities"]), "topics": parse_text_list(r["topics"]),
            "importance": r["importance"], "created_at": r["created_at"],
        })
    db.close()
    return {"memories": memories, "count": len(memories)}


def store_consolidation(
    source_ids: list[int],
    summary: str,
    insight: str,
    connections: list[dict] | str,
) -> dict:
    """Store a consolidation result and mark source memories as consolidated.

    Args:
        source_ids: List of memory IDs that were consolidated.
        summary: A synthesized summary across all source memories.
        insight: One key pattern or insight discovered.
        connections: List of dicts with 'from_id', 'to_id', 'relationship'.

    Returns:
        dict with confirmation.
    """
    source_ids = as_int_list(source_ids)
    connections = as_dict_list(connections)
    db = get_db()
    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO consolidations (source_ids, summary, insight, created_at) VALUES (?, ?, ?, ?)",
        (json.dumps(source_ids), summary, insight, now),
    )
    for conn in connections:
        from_id, to_id = conn.get("from_id"), conn.get("to_id")
        rel = conn.get("relationship", "")
        if from_id and to_id:
            for mid in [from_id, to_id]:
                row = db.execute("SELECT connections FROM memories WHERE id = ?", (mid,)).fetchone()
                if row:
                    existing = parse_json_list(row["connections"])
                    existing.append({"linked_to": to_id if mid == from_id else from_id, "relationship": rel})
                    db.execute("UPDATE memories SET connections = ? WHERE id = ?", (json.dumps(existing), mid))
    if source_ids:
        placeholders = ",".join("?" * len(source_ids))
        db.execute(f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders})", source_ids)
    db.commit()
    db.close()
    log.info(f"🔄 Consolidated {len(source_ids)} memories. Insight: {insight[:80]}...")
    return {"status": "consolidated", "memories_processed": len(source_ids), "insight": insight}


def read_consolidation_history() -> dict:
    """Read past consolidation insights.

    Returns:
        dict with list of consolidation records.
    """
    db = get_db()
    rows = db.execute("SELECT * FROM consolidations ORDER BY created_at DESC LIMIT 10").fetchall()
    result = [{"summary": r["summary"], "insight": r["insight"], "source_ids": r["source_ids"]} for r in rows]
    db.close()
    return {"consolidations": result, "count": len(result)}


def get_memory_stats() -> dict:
    """Get current memory statistics.

    Returns:
        dict with counts of memories, consolidations, etc.
    """
    db = get_db()
    total = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    unconsolidated = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
    consolidations = db.execute("SELECT COUNT(*) as c FROM consolidations").fetchone()["c"]
    db.close()
    return {
        "total_memories": total,
        "unconsolidated": unconsolidated,
        "consolidations": consolidations,
    }


def delete_memory(memory_id: int) -> dict:
    """Delete a memory by ID.

    Args:
        memory_id: The ID of the memory to delete.

    Returns:
        dict with status.
    """
    db = get_db()
    row = db.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone()
    if not row:
        db.close()
        return {"status": "not_found", "memory_id": memory_id}
    db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    db.commit()
    db.close()
    log.info(f"🗑️  Deleted memory #{memory_id}")
    return {"status": "deleted", "memory_id": memory_id}


def clear_all_memories(inbox_path: str | None = None) -> dict:
    """Delete all memories, consolidations, and inbox files. Full reset."""
    db = get_db()
    mem_count = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
    db.execute("DELETE FROM memories")
    db.execute("DELETE FROM consolidations")
    db.execute("DELETE FROM processed_files")
    db.commit()
    db.close()

    # Also clear the inbox folder so files aren't re-ingested
    files_deleted = 0
    if inbox_path:
        folder = Path(inbox_path)
        if folder.is_dir():
            for f in folder.iterdir():
                if f.name.startswith("."):
                    continue  # keep hidden files like .gitkeep
                try:
                    if f.is_file():
                        f.unlink()
                        files_deleted += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        files_deleted += 1
                except OSError as e:
                    log.error(f"Failed to delete {f.name}: {e}")

    log.info(f"🗑️  Cleared all {mem_count} memories, deleted {files_deleted} inbox files")
    return {"status": "cleared", "memories_deleted": mem_count, "files_deleted": files_deleted}


# ─── ADK Agents ────────────────────────────────────────────────


class MemoryAgents(NamedTuple):
    ingest: Agent
    consolidate: Agent
    query: Agent
    orchestrator: Agent


def build_model(
    provider: str | None = None,
    name: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> str | BaseLlm:
    """Return the ADK model binding for a provider.

    LM Studio and Ollama are reached through ADK's LiteLlm adapter; LM Studio
    serves an OpenAI-compatible API, Ollama its native chat API. Any other
    provider name is treated as a Gemini model name. Missing arguments come from
    the effective selection (dashboard setting, then env vars, then defaults).
    """
    if provider is None or name is None:
        selected_provider, selected_name = config.load_selection()
        provider = provider or selected_provider
        name = name or selected_name

    if provider not in {"lmstudio", "ollama"}:
        return name

    from google.adk.models.lite_llm import LiteLlm

    if provider == "ollama":
        return LiteLlm(
            model=f"ollama_chat/{name}",
            api_base=api_base or OLLAMA_API_BASE,
        )
    return LiteLlm(
        model=f"openai/{name}",
        api_base=api_base or LMSTUDIO_API_BASE,
        api_key=api_key or LMSTUDIO_API_KEY,
    )


def build_agents(model: str | BaseLlm | None = None) -> MemoryAgents:
    model = build_model() if model is None else model
    ingest_agent = Agent(
        name="ingest_agent",
        model=model,
        description="Processes raw text or media into structured memory. Call this when new information arrives.",
        instruction=(
            "You are a Memory Ingest Agent. You handle ALL types of input — text, images,\n"
            "audio, video, and PDFs. For any input you receive:\n"
            "1. Thoroughly describe what the content contains\n"
            "2. Create a concise 1-2 sentence summary\n"
            "3. Extract key entities (people, companies, products, concepts, objects, locations)\n"
            "4. Assign 2-4 topic tags\n"
            "5. Rate importance from 0.0 to 1.0\n"
            "6. Call store_memory with all extracted information\n\n"
            "For images: describe the scene, objects, text, people, and any visual details.\n"
            "For audio/video: describe the spoken content, sounds, scenes, and key moments.\n"
            "For PDFs: extract and summarize the document content.\n\n"
            "Use the full description as raw_text in store_memory so the context is preserved.\n"
            "Always call store_memory. Be concise and accurate.\n"
            "After storing, confirm what was stored in one sentence."
        ),
        tools=[store_memory],
    )

    consolidate_agent = Agent(
        name="consolidate_agent",
        model=model,
        description="Merges related memories and finds patterns. Call this periodically.",
        instruction=(
            "You are a Memory Consolidation Agent. You:\n"
            "1. Call read_unconsolidated_memories to see what needs processing\n"
            "2. If fewer than 2 memories, say nothing to consolidate\n"
            "3. Find connections and patterns across the memories\n"
            "4. Create a synthesized summary and one key insight\n"
            "5. Call store_consolidation with source_ids, summary, insight, and connections\n\n"
            "Connections: list of dicts with 'from_id', 'to_id', 'relationship' keys.\n"
            "Think deeply about cross-cutting patterns."
        ),
        tools=[read_unconsolidated_memories, store_consolidation],
    )

    query_agent = Agent(
        name="query_agent",
        model=model,
        description="Answers questions using stored memories.",
        instruction=(
            "You are a Memory Query Agent. When asked a question:\n"
            "1. Call read_all_memories to access the memory store\n"
            "2. Call read_consolidation_history for higher-level insights\n"
            "3. Synthesize an answer based ONLY on stored memories\n"
            "4. Reference memory IDs: [Memory 1], [Memory 2], etc.\n"
            "5. If no relevant memories exist, say so honestly\n\n"
            "Be thorough but concise. Always cite sources."
        ),
        tools=[read_all_memories, read_consolidation_history],
    )

    orchestrator = Agent(
        name="memory_orchestrator",
        model=model,
        description="Routes memory operations to specialist agents.",
        instruction=(
            "You are the Memory Orchestrator for an always-on memory system.\n"
            "Route requests to the right sub-agent:\n"
            "- New information -> ingest_agent\n"
            "- Consolidation request -> consolidate_agent\n"
            "- Questions -> query_agent\n"
            "- Status check -> call get_memory_stats and report\n\n"
            "After the sub-agent completes, give a brief summary."
        ),
        sub_agents=[ingest_agent, consolidate_agent, query_agent],
        tools=[get_memory_stats],
    )

    return MemoryAgents(
        ingest=ingest_agent,
        consolidate=consolidate_agent,
        query=query_agent,
        orchestrator=orchestrator,
    )


def fallback_summary(text: str, maximum: int = 200) -> str:
    collapsed = " ".join(text.split())
    return collapsed[:maximum] + ("..." if len(collapsed) > maximum else "")


# ─── Agent Runner ──────────────────────────────────────────────


class MemoryAgent:
    def __init__(self, model: str | BaseLlm | None = None) -> None:
        self.model = build_model() if model is None else model
        self.agents = build_agents(self.model)
        self.session_service = InMemorySessionService()
        self._runners = {
            "ingest": self._make_runner("ingest", self.agents.ingest),
            "consolidate": self._make_runner("consolidate", self.agents.consolidate),
            "query": self._make_runner("query", self.agents.query),
        }

    def _make_runner(self, name: str, agent: Agent) -> Runner:
        return Runner(
            agent=agent,
            app_name=f"memory_{name}",
            session_service=self.session_service,
        )

    async def _run(self, name: str, message: str) -> str:
        return await self._run_parts(name, [types.Part.from_text(text=message)])

    async def _run_parts(self, name: str, parts: list[types.Part]) -> str:
        """Run one specialist agent directly; no orchestrator routing involved."""
        runner = self._runners[name]
        session = await self.session_service.create_session(
            app_name=f"memory_{name}", user_id="agent",
        )
        content = types.Content(role="user", parts=parts)
        return await self._execute(runner, session, content)

    async def _execute(self, runner: Runner, session, content: types.Content) -> str:
        """Run the agent with the given content and return the text response."""
        response = ""
        async for event in runner.run_async(
            user_id="agent", session_id=session.id, new_message=content,
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        response += part.text
        return response

    async def ingest(self, text: str, source: str = "") -> str:
        before = get_memory_stats()["total_memories"]
        msg = f"Remember this information (source: {source}):\n\n{text}" if source else f"Remember this information:\n\n{text}"
        reply = await self._run("ingest", msg)
        if get_memory_stats()["total_memories"] > before:
            return reply.strip() or "Memory stored."

        stored = store_memory(
            raw_text=text,
            summary=fallback_summary(text),
            entities=[],
            topics=[],
            importance=0.5,
            source=source,
        )
        log.warning(
            f"⚠️  Model returned no store_memory call; saved memory #{stored['memory_id']} verbatim"
        )
        reason = (
            f"Memory #{stored['memory_id']} saved verbatim without entity/topic extraction: "
            f"the model returned no store_memory call. Use a stronger model for extraction."
        )
        return f"{reply.strip()}\n\n{reason}" if reply.strip() else reason

    async def ingest_file(self, file_path: Path) -> str:
        """Ingest a media file (image, audio, video, PDF) via multimodal."""
        if PROVIDER != "gemini":
            log.warning(
                f"⚠️  Skipping {file_path.name} — media ingestion requires a Gemini model "
                f"(MEMORY_PROVIDER is '{PROVIDER}')"
            )
            return f"Skipped: media ingestion requires a Gemini model (provider: {PROVIDER})"

        suffix = file_path.suffix.lower()
        mime_type = MEDIA_EXTENSIONS.get(suffix)
        if not mime_type:
            # Fallback to mimetypes module
            mime_type, _ = mimetypes.guess_type(str(file_path))
            mime_type = mime_type or "application/octet-stream"

        file_bytes = file_path.read_bytes()
        size_mb = len(file_bytes) / (1024 * 1024)

        # Gemini has a ~20MB inline limit; skip very large files
        if size_mb > 20:
            log.warning(f"⚠️  Skipping {file_path.name} ({size_mb:.1f}MB) — exceeds 20MB limit")
            return f"Skipped: file too large ({size_mb:.1f}MB)"

        prompt = (
            f"Remember this file (source: {file_path.name}, type: {mime_type}).\n\n"
            f"Thoroughly analyze the content of this {mime_type.split('/')[0]} file and "
            f"extract all meaningful information for memory storage."
        )
        log.info(f"🔮 Ingesting {mime_type.split('/')[0]}: {file_path.name} ({size_mb:.1f}MB)")
        parts = [
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
        ]
        return await self._run_parts("ingest", parts)

    async def consolidate(self) -> str:
        pending = get_memory_stats()["unconsolidated"]
        if pending < 2:
            return (
                f"Nothing to consolidate: {pending} unconsolidated "
                f"memor{'y' if pending == 1 else 'ies'} (at least 2 are needed). "
                f"Ingest more memories first."
            )
        return await self._run(
            "consolidate", "Consolidate unconsolidated memories. Find connections and patterns."
        )

    async def query(self, question: str) -> str:
        return await self._run("query", f"Based on my memories, answer: {question}")

    async def status(self) -> str:
        stats = get_memory_stats()
        return (
            f"Memories: {stats['total_memories']}, "
            f"unconsolidated: {stats['unconsolidated']}, "
            f"consolidations: {stats['consolidations']}"
        )


# ─── File Watcher ──────────────────────────────────────────────


async def watch_folder(agent: MemoryAgent, folder: Path, poll_interval: int = 5):
    """Watch a folder for new files and ingest them (text, images, audio, video, PDFs)."""
    folder.mkdir(parents=True, exist_ok=True)
    db = get_db()
    log.info(f"👁️  Watching: {folder}/  (supports: text, images, audio, video, PDFs)")

    while True:
        try:
            for f in sorted(folder.iterdir()):
                if f.name.startswith("."):
                    continue  # skip hidden files
                suffix = f.suffix.lower()
                if suffix not in ALL_SUPPORTED:
                    continue
                row = db.execute("SELECT 1 FROM processed_files WHERE path = ?", (str(f),)).fetchone()
                if row:
                    continue

                try:
                    if suffix in TEXT_EXTENSIONS:
                        # Text-based files — read as string
                        log.info(f"📄 New text file: {f.name}")
                        text = f.read_text(encoding="utf-8", errors="replace")[:10000]
                        if text.strip():
                            await agent.ingest(text, source=f.name)
                    else:
                        # Media files — send as multimodal bytes
                        log.info(f"🖼️  New media file: {f.name}")
                        await agent.ingest_file(f)
                except Exception as file_err:
                    log.error(f"Error ingesting {f.name}: {file_err}")

                db.execute(
                    "INSERT INTO processed_files (path, processed_at) VALUES (?, ?)",
                    (str(f), datetime.now(timezone.utc).isoformat()),
                )
                db.commit()
        except Exception as e:
            log.error(f"Watch error: {e}")

        await asyncio.sleep(poll_interval)


# ─── Consolidation Timer ──────────────────────────────────────


async def consolidation_loop(agent: MemoryAgent, interval_minutes: int = 30):
    """Run consolidation periodically, like sleep cycles."""
    log.info(f"🔄 Consolidation: every {interval_minutes} minutes")
    while True:
        await asyncio.sleep(interval_minutes * 60)
        try:
            db = get_db()
            count = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
            db.close()
            if count >= 2:
                log.info(f"🔄 Running consolidation ({count} unconsolidated memories)...")
                result = await agent.consolidate()
                log.info(f"🔄 {result[:100]}")
            else:
                log.info(f"🔄 Skipping consolidation ({count} unconsolidated memories)")
        except Exception as e:
            log.error(f"Consolidation error: {e}")


# ─── HTTP API ──────────────────────────────────────────────────


def build_http(agent: MemoryAgent, watch_path: str = "./inbox"):
    app = web.Application()

    async def handle_query(request: web.Request):
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"error": "missing ?q= parameter"}, status=400)
        answer = await agent.query(q)
        return web.json_response({"question": q, "answer": answer})

    async def handle_ingest(request: web.Request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)
        source = data.get("source", "api")
        result = await agent.ingest(text, source=source)
        return web.json_response({"status": "ingested", "response": result})

    async def handle_consolidate(request: web.Request):
        result = await agent.consolidate()
        return web.json_response({"status": "done", "response": result})

    async def handle_status(request: web.Request):
        stats = get_memory_stats()
        return web.json_response(stats)

    async def handle_memories(request: web.Request):
        data = read_all_memories()
        return web.json_response(data)

    async def handle_delete(request: web.Request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        memory_id = data.get("memory_id")
        if not memory_id:
            return web.json_response({"error": "missing 'memory_id' field"}, status=400)
        result = delete_memory(int(memory_id))
        return web.json_response(result)

    async def handle_clear(request: web.Request):
        result = clear_all_memories(inbox_path=watch_path)
        return web.json_response(result)

    app.router.add_get("/query", handle_query)
    app.router.add_post("/ingest", handle_ingest)
    app.router.add_post("/consolidate", handle_consolidate)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/memories", handle_memories)
    app.router.add_post("/delete", handle_delete)
    app.router.add_post("/clear", handle_clear)

    return app


# ─── Main ──────────────────────────────────────────────────────


async def main_async(args):
    agent = MemoryAgent()

    log.info("🧠 Agent Memory Layer starting")
    provider, model = config.load_selection()
    endpoint = {"lmstudio": LMSTUDIO_API_BASE, "ollama": OLLAMA_API_BASE}.get(provider, "google")
    log.info(f"   Model: {model} (provider: {provider}, endpoint: {endpoint})")
    log.info(f"   Database: {DB_PATH}")
    log.info(f"   Watch: {args.watch}")
    log.info(f"   Consolidate: every {args.consolidate_every}m")
    log.info(f"   API: http://localhost:{args.port}")
    log.info("")

    # Start background tasks
    tasks = [
        asyncio.create_task(watch_folder(agent, Path(args.watch))),
        asyncio.create_task(consolidation_loop(agent, args.consolidate_every)),
    ]

    # Start HTTP server
    app = build_http(agent, watch_path=args.watch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log.info(f"✅ Agent running. Drop files in {args.watch}/ or POST to http://localhost:{args.port}/ingest")
    log.info(f"   Supported: text, images, audio, video, PDFs")
    log.info("")

    # Wait forever
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Agent Memory Layer - Always-On ADK Agent")
    parser.add_argument("--watch", default="./inbox", help="Folder to watch for new files (default: ./inbox)")
    parser.add_argument("--port", type=int, default=8888, help="HTTP API port (default: 8888)")
    parser.add_argument("--consolidate-every", type=int, default=30, help="Consolidation interval in minutes (default: 30)")
    args = parser.parse_args()

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def shutdown(sig):
        log.info(f"\n👋 Shutting down (signal {sig})...")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown, sig)

    try:
        loop.run_until_complete(main_async(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()
        log.info("🧠 Agent stopped.")


if __name__ == "__main__":
    main()
