from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import dashboard
import server
from dashboard import read_dashboard_data
from memory import agent
from streamlit.testing.v1 import AppTest

DASHBOARD_SCRIPT = Path(__file__).resolve().parent.parent / "dashboard.py"


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


class DashboardAppTests(unittest.TestCase):
    """Render the Streamlit script itself so widget calls are exercised."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "memory.db"
        self.previous_db_path = agent.DB_PATH
        agent.DB_PATH = str(self.db_path)
        db = agent.get_db()
        db.close()
        agent.store_memory(
            raw_text="The rollout is planned for Friday.",
            summary="Rollout planned for Friday.",
            entities=["Rollout"],
            topics=["planning"],
            importance=0.8,
            source="plan.md",
        )
        self.env_patch = mock.patch.dict(os.environ, {"MEMORY_DB": str(self.db_path)})
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        agent.DB_PATH = self.previous_db_path
        self.temp_dir.cleanup()

    def _run_app(self) -> AppTest:
        app = AppTest.from_file(str(DASHBOARD_SCRIPT), default_timeout=60)
        app.run()
        return app

    def test_renders_and_filters_by_consolidation_status(self) -> None:
        app = self._run_app()

        self.assertEqual([str(error.value) for error in app.exception], [])
        self.assertEqual([metric.value for metric in app.metric], ["1", "1", "0"])
        self.assertEqual(app.radio[0].label, "Consolidation status")
        self.assertEqual(app.radio[0].value, "All")

        app.radio[0].set_value("Consolidated").run()

        self.assertEqual([str(error.value) for error in app.exception], [])
        self.assertEqual(app.subheader[0].value, "Recent memories (0 of 1 shown)")

    def test_shows_startup_hint_while_the_database_is_missing(self) -> None:
        missing = Path(self.temp_dir.name) / "not-created" / "memory.db"
        os.environ["MEMORY_DB"] = str(missing)

        app = self._run_app()

        self.assertEqual([str(error.value) for error in app.exception], [])
        self.assertFalse(missing.exists())
        self.assertEqual(len(app.info), 1)
        self.assertIn(str(missing), app.info[0].value)
        self.assertIn("server.py", app.code[0].value)


class DatabasePathTests(unittest.TestCase):
    def test_entrypoints_share_an_absolute_default_database_path(self) -> None:
        self.assertTrue(dashboard.DB_PATH.is_absolute())
        self.assertEqual(dashboard.DB_PATH, server.DB_PATH)


if __name__ == "__main__":
    unittest.main()
