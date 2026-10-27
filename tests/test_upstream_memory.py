from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from memory import agent


class UpstreamMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_db_path = agent.DB_PATH
        agent.DB_PATH = str(Path(self.temp_dir.name) / "memory.db")
        db = agent.get_db()
        db.close()

    def tearDown(self) -> None:
        agent.DB_PATH = self.previous_db_path
        self.temp_dir.cleanup()

    def test_upstream_store_and_read_functions_share_memory_data(self) -> None:
        stored = agent.store_memory(
            raw_text="Release review is scheduled for Friday.",
            summary="Release review on Friday.",
            entities=["Release"],
            topics=["planning"],
            importance=0.8,
            source="planning.md",
        )

        memories = agent.read_all_memories()
        stats = agent.get_memory_stats()

        self.assertEqual(stored["status"], "stored")
        self.assertEqual(memories["memories"][0]["id"], stored["memory_id"])
        self.assertEqual(memories["memories"][0]["source"], "planning.md")
        self.assertEqual(memories["memories"][0]["summary"], "Release review on Friday.")
        self.assertEqual(stats["total_memories"], 1)


if __name__ == "__main__":
    unittest.main()
