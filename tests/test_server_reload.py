from __future__ import annotations

import unittest
from unittest import mock

import server


class MemoryAgentReloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_state = (server._memory_agent, server._memory_selection)
        server._memory_agent = None
        server._memory_selection = None

    def tearDown(self) -> None:
        server._memory_agent, server._memory_selection = self.previous_state

    def test_rebuilds_the_agent_when_the_selection_changes(self) -> None:
        created: list[object] = []

        def fake_agent(model: object) -> mock.Mock:
            created.append(model)
            return mock.Mock(name="memory_agent")

        selections = [("lmstudio", "a-model"), ("lmstudio", "a-model"), ("ollama", "b-model")]
        with (
            mock.patch.object(server.agent, "MemoryAgent", side_effect=fake_agent),
            mock.patch.object(server, "load_selection", side_effect=selections),
        ):
            first = server._get_memory_agent()
            second = server._get_memory_agent()
            third = server._get_memory_agent()

        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(
            [getattr(model, "model") for model in created],
            ["openai/a-model", "ollama_chat/b-model"],
        )


if __name__ == "__main__":
    unittest.main()
