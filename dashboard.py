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

DB_PATH = Path(os.getenv("MEMORY_DB", Path(__file__).with_name("memory.db")))
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
        for field in ("entities", "topics", "connections"):
            memory[field] = json.loads(memory[field])
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


def render_dashboard() -> None:
    try:
        data = read_dashboard_data()
    except (sqlite3.Error, json.JSONDecodeError) as error:
        st.error(f"Cannot read memory database at {DB_PATH}: {error}")
        return

    st.caption(f"Read-only database monitor · `{DB_PATH}` · refreshes every 15 seconds")
    if data is None:
        st.info("Waiting for the MCP server to create the memory database.")
        return

    stats = data["stats"]
    total, pending, consolidated = st.columns(3)
    total.metric("Memories", stats["total_memories"])
    pending.metric("Pending consolidation", stats["unconsolidated"])
    consolidated.metric("Consolidations", stats["consolidations"])

    query = st.text_input("Filter recent memories", placeholder="Search summary, source, topic, or entity")
    status_filter = st.selectbox(
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
        "Monitor the shared MCP memory store. This dashboard is read-only and "
        "does not call Gemini or alter stored memories."
    )
    render_monitor()


if __name__ == "__main__":
    main()
