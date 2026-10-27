from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from memory import agent

MODEL = "gemini-3.1-flash-lite"


class MemoryAgentTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_db_path = agent.DB_PATH
        agent.DB_PATH = str(Path(self.temp_dir.name) / "memory.db")
        db = agent.get_db()
        db.close()
        self.memory_agent = agent.MemoryAgent(MODEL)

    def tearDown(self) -> None:
        agent.DB_PATH = self.previous_db_path
        self.temp_dir.cleanup()

    def stored_memories(self) -> list[dict]:
        return agent.read_all_memories()["memories"]


class IngestTests(MemoryAgentTestCase):
    def test_uses_the_ingest_agent_directly(self) -> None:
        with mock.patch.object(self.memory_agent, "_run", new=mock.AsyncMock(return_value="Stored.")) as run:
            asyncio.run(self.memory_agent.ingest("The launch moved to Thursday."))

        self.assertEqual(run.await_args.args[0], "ingest")

    def test_stores_verbatim_when_the_model_does_not_call_store_memory(self) -> None:
        with mock.patch.object(self.memory_agent, "_run", new=mock.AsyncMock(return_value="Nothing to consolidate.")):
            reply = asyncio.run(self.memory_agent.ingest("The launch moved to Thursday.", source="notes.md"))

        memories = self.stored_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["summary"], "The launch moved to Thursday.")
        self.assertEqual(memories[0]["source"], "notes.md")
        self.assertIn("saved verbatim", reply)
        self.assertIn("Nothing to consolidate.", reply)

    def test_reports_the_stored_memory_when_the_model_stores_it(self) -> None:
        async def fake_run(name: str, message: str) -> str:
            agent.store_memory(
                raw_text="The launch moved to Thursday.",
                summary="Launch moved to Thursday.",
                entities=["Launch"],
                topics=["planning"],
                importance=0.7,
                source="notes.md",
            )
            return "Stored memory #1."

        with mock.patch.object(self.memory_agent, "_run", new=fake_run):
            reply = asyncio.run(self.memory_agent.ingest("The launch moved to Thursday."))

        memories = self.stored_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(reply, "Stored memory #1.")
        self.assertNotIn("verbatim", reply)

    def test_fallback_summary_is_collapsed_and_truncated(self) -> None:
        long_text = "word " * 100

        with mock.patch.object(self.memory_agent, "_run", new=mock.AsyncMock(return_value="")):
            asyncio.run(self.memory_agent.ingest(long_text))

        summary = self.stored_memories()[0]["summary"]
        self.assertEqual(len(summary), 203)
        self.assertFalse(summary.endswith(" "))
        self.assertTrue(summary.endswith("..."))


class ConsolidateGuardTests(MemoryAgentTestCase):
    def test_refuses_to_run_with_fewer_than_two_unconsolidated_memories(self) -> None:
        agent.store_memory(
            raw_text="Only one.",
            summary="Only one.",
            entities=[],
            topics=[],
            importance=0.5,
        )

        with mock.patch.object(self.memory_agent, "_run", new=mock.AsyncMock()) as run:
            reply = asyncio.run(self.memory_agent.consolidate())

        run.assert_not_awaited()
        self.assertIn("Nothing to consolidate: 1 unconsolidated memory", reply)

    def test_runs_the_consolidate_agent_with_two_pending_memories(self) -> None:
        for index in range(2):
            agent.store_memory(
                raw_text=f"Fact {index}.",
                summary=f"Fact {index}.",
                entities=[],
                topics=[],
                importance=0.5,
            )

        with mock.patch.object(self.memory_agent, "_run", new=mock.AsyncMock(return_value="Insight.")) as run:
            reply = asyncio.run(self.memory_agent.consolidate())

        self.assertEqual(run.await_args.args[0], "consolidate")
        self.assertEqual(reply, "Insight.")


class StatusTests(MemoryAgentTestCase):
    def test_status_is_deterministic(self) -> None:
        agent.store_memory(
            raw_text="Fact.",
            summary="Fact.",
            entities=[],
            topics=[],
            importance=0.5,
        )

        reply = asyncio.run(self.memory_agent.status())

        self.assertEqual(reply, "Memories: 1, unconsolidated: 1, consolidations: 0")


class TextListNormalisationTests(MemoryAgentTestCase):
    def test_string_entities_and_topics_are_stored_as_lists(self) -> None:
        agent.store_memory(
            raw_text="Fact.",
            summary="Fact.",
            entities="release schedule, QA pass",
            topics="planning",
            importance=0.5,
        )

        memory = self.stored_memories()[0]
        self.assertEqual(memory["entities"], ["release schedule", "QA pass"])
        self.assertEqual(memory["topics"], ["planning"])

    def test_rows_written_as_plain_strings_are_normalised_on_read(self) -> None:
        db = agent.get_db()
        db.execute(
            """INSERT INTO memories (source, raw_text, summary, entities, topics, importance, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            ("src.md", "Raw.", "Summary.", json.dumps("a, b"), json.dumps("x"), 0.5, "2026-01-01T00:00:00+00:00"),
        )
        db.commit()
        db.close()

        memory = self.stored_memories()[0]
        self.assertEqual(memory["entities"], ["a", "b"])
        self.assertEqual(memory["topics"], ["x"])


class ConsolidationStorageTests(MemoryAgentTestCase):
    def store_two_memories(self) -> None:
        for index in range(2):
            agent.store_memory(
                raw_text=f"Fact {index}.",
                summary=f"Fact {index}.",
                entities=[],
                topics=[],
                importance=0.5,
            )

    def test_accepts_string_connections_and_string_source_ids(self) -> None:
        self.store_two_memories()

        result = agent.store_consolidation(
            source_ids="[1, 2]",
            summary="Both facts relate.",
            insight="Pattern.",
            connections='[{"from_id": 1, "to_id": 2, "relationship": "related"}]',
        )

        memories = self.stored_memories()
        self.assertEqual(result["memories_processed"], 2)
        self.assertTrue(all(memory["consolidated"] for memory in memories))
        self.assertEqual(
            sorted((memory["id"], memory["connections"][0]["linked_to"]) for memory in memories),
            [(1, 2), (2, 1)],
        )

    def test_still_records_the_consolidation_when_connections_are_unusable(self) -> None:
        self.store_two_memories()

        result = agent.store_consolidation(
            source_ids="1,2",
            summary="Summary.",
            insight="Insight.",
            connections="not a list",
        )

        self.assertEqual(result["memories_processed"], 2)
        self.assertTrue(all(memory["consolidated"] for memory in self.stored_memories()))


if __name__ == "__main__":
    unittest.main()
