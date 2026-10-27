from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dashboard import read_dashboard_data
from upstream import agent


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.db"
        self.previous_db_path = agent.DB_PATH
        agent.DB_PATH = str(self.db_path)
        db = agent.get_db()
        db.close()

    def tearDown(self) -> None:
        agent.DB_PATH = self.previous_db_path
        self.temp_dir.cleanup()

    def test_reads_stats_memories_and_consolidations(self) -> None:
        stored = agent.store_memory(
            raw_text="The rollout is planned for Friday.",
            summary="Rollout planned for Friday.",
            entities=["Rollout"],
            topics=["planning"],
            importance=0.8,
            source="plan.md",
        )

        data = read_dashboard_data(self.db_path)

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["stats"]["total_memories"], 1)
        self.assertEqual(data["stats"]["unconsolidated"], 1)
        self.assertEqual(data["memories"][0]["id"], stored["memory_id"])
        self.assertEqual(data["memories"][0]["topics"], ["planning"])
        self.assertEqual(data["consolidations"], [])

    def test_returns_none_without_creating_a_missing_database(self) -> None:
        missing_path = Path(self.temp_dir.name) / "not-created" / "memory.db"

        self.assertIsNone(read_dashboard_data(missing_path))
        self.assertFalse(missing_path.exists())


if __name__ == "__main__":
    unittest.main()
