"""Read-only Streamlit monitor adapted from the upstream memory dashboard.

Upstream source: GoogleCloudPlatform/generative-ai, pinned to
35672474133c83fbcc9f97876fafad4c108822f2. The upstream dashboard uses the
agent's REST API; this version reads the shared MCP SQLite database instead.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import streamlit as st

from asyncio_noise import quiet_windows_asyncio_resets
from memory.config import (
    PROVIDERS,
    config_path,
    default_model,
    discover_models,
    load_selection,
    save_selection,
)
from memory.records import parse_json_list, parse_text_list

quiet_windows_asyncio_resets()

DB_PATH = Path(os.getenv("MEMORY_DB", Path(__file__).resolve().with_name("memory.db")))
MAX_RECENT_MEMORIES = 50
MAX_RECENT_CONSOLIDATIONS = 20


def read_dashboard_data(db_path: Path = DB_PATH) -> dict[str, Any] | None:
    """Read monitoring data without creating or changing the database."""
    if not db_path.is_file():
        return None

    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        stats = {
            "total_memories": connection.execute(
                "SELECT COUNT(*) FROM memories"
            ).fetchone()[0],
            "unconsolidated": connection.execute(
                "SELECT COUNT(*) FROM memories WHERE consolidated = 0"
            ).fetchone()[0],
            "consolidations": connection.execute(
                "SELECT COUNT(*) FROM consolidations"
            ).fetchone()[0],
        }
        memory_rows = connection.execute(
            """SELECT id, source, summary, entities, topics, connections,
                      importance, created_at, consolidated
               FROM memories
               ORDER BY created_at DESC, id DESC
               LIMIT ?""",
            (MAX_RECENT_MEMORIES,),
        ).fetchall()
        consolidation_rows = connection.execute(
            """SELECT id, source_ids, summary, insight, created_at
               FROM consolidations
               ORDER BY created_at DESC, id DESC
               LIMIT ?""",
            (MAX_RECENT_CONSOLIDATIONS,),
        ).fetchall()
    finally:
        connection.close()

    memories = []
    for row in memory_rows:
        memory = dict(row)
        memory["entities"] = parse_text_list(memory["entities"])
        memory["topics"] = parse_text_list(memory["topics"])
        memory["connections"] = parse_json_list(memory["connections"])
        memory["consolidated"] = bool(memory["consolidated"])
        memories.append(memory)

    consolidations = []
    for row in consolidation_rows:
        consolidation = dict(row)
        consolidation["source_ids"] = json.loads(consolidation["source_ids"])
        consolidations.append(consolidation)

    return {
        "stats": stats,
        "memories": memories,
        "consolidations": consolidations,
    }


def render_memory(memory: dict[str, Any]) -> None:
    created_at = memory["created_at"].replace("T", " ")[:19]
    with st.container(border=True):
        heading, status = st.columns([4, 1])
        heading.markdown(f"**Memory #{memory['id']}** · {created_at}")
        status.caption("Consolidated" if memory["consolidated"] else "Pending")
        if memory["source"]:
            st.caption(f"Source: {memory['source']}")
        st.write(memory["summary"])
        details = []
        if memory["topics"]:
            details.append("Topics: " + ", ".join(memory["topics"]))
        if memory["entities"]:
            details.append("Entities: " + ", ".join(memory["entities"][:8]))
        if memory["connections"]:
            details.append(f"Connections: {len(memory['connections'])}")
        if details:
            st.caption(" · ".join(details))
        st.progress(max(0.0, min(float(memory["importance"]), 1.0)))


@st.cache_data(ttl=15, show_spinner=False)
def available_models(provider: str) -> list[str]:
    return discover_models(provider)


def save_selection_from_widgets(provider_key: str, model_key_prefix: str) -> None:
    """Persist the pickers when the user changes one; widget resets never write."""
    provider = st.session_state.get(provider_key)
    if provider not in PROVIDERS:
        return
    model = st.session_state.get(f"{model_key_prefix}-{provider}") or default_model(provider)
    save_selection(provider, model)


def render_model_settings() -> None:
    """Provider and model pickers; the server re-reads them on its next tool call."""
    saved_provider, saved_model = load_selection()

    st.selectbox(
        "API provider",
        PROVIDERS,
        index=PROVIDERS.index(saved_provider) if saved_provider in PROVIDERS else 0,
        key="model-provider",
        on_change=save_selection_from_widgets,
        args=("model-provider", "model-name"),
    )
    provider = st.session_state["model-provider"]
    fallback = saved_model if provider == saved_provider else default_model(provider)
    options = list(dict.fromkeys([fallback, *available_models(provider)]))
    st.selectbox(
        "Model",
        options,
        index=options.index(fallback) if fallback in options else 0,
        accept_new_options=True,
        key=f"model-name-{provider}",
        on_change=save_selection_from_widgets,
        args=("model-provider", "model-name"),
        help="Model ids are read from the running local server; type a name to use one that is not listed.",
    )

    st.caption(
        f"Applies to the MCP server on its next tool call · saved in `{config_path()}`"
    )
    st.divider()


def render_dashboard() -> None:
    render_model_settings()

    try:
        data = read_dashboard_data()
    except (sqlite3.Error, json.JSONDecodeError) as error:
        st.error(f"Cannot read memory database at {DB_PATH}: {error}")
        return

    st.caption(f"Read-only database monitor · `{DB_PATH}` · refreshes every 15 seconds")
    if data is None:
        st.info(f"Waiting for the MCP server to create the memory database at `{DB_PATH}`.")
        st.code("uv run python server.py", language="powershell")
        st.caption(
            "The dashboard only reads the database; the server creates it on startup "
            "and appends memories as tools are called."
        )
        return

    stats = data["stats"]
    total, pending, consolidated = st.columns(3)
    total.metric("Memories", stats["total_memories"])
    pending.metric("Pending consolidation", stats["unconsolidated"])
    consolidated.metric("Consolidations", stats["consolidations"])

    query = st.text_input("Filter recent memories", placeholder="Search summary, source, topic, or entity")
    status_filter = st.radio(
        "Consolidation status",
        ("All", "Pending", "Consolidated"),
        horizontal=True,
    )
    matching_memories = data["memories"]
    if query.strip():
        needle = query.casefold().strip()
        matching_memories = [
            memory
            for memory in matching_memories
            if needle
            in " ".join(
                [
                    memory["source"],
                    memory["summary"],
                    *memory["topics"],
                    *memory["entities"],
                ]
            ).casefold()
        ]
    if status_filter != "All":
        is_consolidated = status_filter == "Consolidated"
        matching_memories = [
            memory for memory in matching_memories
            if memory["consolidated"] is is_consolidated
        ]

    st.subheader(f"Recent memories ({len(matching_memories)} of {stats['total_memories']} shown)")
    if matching_memories:
        for memory in matching_memories:
            render_memory(memory)
    else:
        st.info("No memories match these filters.")

    st.subheader("Recent consolidation history")
    if data["consolidations"]:
        for consolidation in data["consolidations"]:
            timestamp = consolidation["created_at"].replace("T", " ")[:19]
            with st.expander(f"#{consolidation['id']} · {timestamp}"):
                st.markdown("**Insight**")
                st.write(consolidation["insight"])
                st.markdown("**Summary**")
                st.write(consolidation["summary"])
                st.caption(f"Source memory IDs: {consolidation['source_ids']}")
    else:
        st.info("No consolidation runs recorded yet.")


@st.fragment(run_every="15s")
def render_monitor() -> None:
    render_dashboard()


def main() -> None:
    st.set_page_config(
        page_title="Agent Memory Monitor",
        page_icon="🧠",
        layout="wide",
    )
    st.title("Agent Memory Monitor")
    st.markdown(
        "Monitor the shared MCP memory store. Stored memories are never modified "
        "here; the pickers below only choose which provider and model the MCP "
        "server uses."
    )
    render_monitor()


if __name__ == "__main__":
    main()
