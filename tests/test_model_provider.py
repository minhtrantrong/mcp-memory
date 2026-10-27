from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from google.adk.models.base_llm import BaseLlm
from memory import agent


class ModelProviderTests(unittest.TestCase):
    def test_lm_studio_is_the_default_provider(self) -> None:
        self.assertEqual(agent.PROVIDER, "lmstudio")
        self.assertEqual(agent.MODEL, "liquid/lfm2.5-1.2b")

    def test_lm_studio_models_use_the_openai_compatible_api(self) -> None:
        model = agent.build_model("lmstudio", "qwen3-0.6b", "http://127.0.0.1:1234/v1")

        self.assertIsInstance(model, BaseLlm)
        self.assertEqual(model.model, "openai/qwen3-0.6b")
        self.assertEqual(model._additional_args["api_base"], "http://127.0.0.1:1234/v1")
        self.assertEqual(model._additional_args["api_key"], "lm-studio")

    def test_ollama_models_are_served_through_lite_llm(self) -> None:
        model = agent.build_model("ollama", "qwen2.5:1.6b", "http://127.0.0.1:11434")

        self.assertIsInstance(model, BaseLlm)
        self.assertEqual(model.model, "ollama_chat/qwen2.5:1.6b")
        self.assertEqual(model._additional_args["api_base"], "http://127.0.0.1:11434")

    def test_gemini_provider_keeps_the_model_name(self) -> None:
        self.assertEqual(
            agent.build_model("gemini", "gemini-3.1-flash-lite"),
            "gemini-3.1-flash-lite",
        )

    def test_build_agents_share_the_configured_model(self) -> None:
        model = agent.build_model("lmstudio", "liquid/lfm2.5-1.2b")

        orchestrator = agent.build_agents(model)

        self.assertEqual(orchestrator.model.model, "openai/liquid/lfm2.5-1.2b")
        self.assertEqual(orchestrator.model._additional_args["api_base"], agent.LMSTUDIO_API_BASE)
        for sub_agent in orchestrator.sub_agents:
            self.assertIs(sub_agent.model, model)

    def test_media_ingestion_is_skipped_without_a_gemini_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            media = Path(temp_dir) / "photo.png"
            media.write_bytes(b"not-a-real-png")

            result = asyncio.run(agent.MemoryAgent().ingest_file(media))

        self.assertTrue(result.startswith("Skipped"))


class LiteLlmCompatibilityTests(unittest.TestCase):
    def test_litellm_tool_path_imports_with_the_pinned_mcp(self) -> None:
        # litellm loads this MCP gateway module whenever tools are passed; releases
        # >=1.106 need mcp 2.x, which conflicts with the FastMCP 1.x pin here.
        from litellm.responses.mcp import chat_completions_handler  # noqa: F401

        self.assertTrue(chat_completions_handler.__name__.endswith("chat_completions_handler"))


if __name__ == "__main__":
    unittest.main()
