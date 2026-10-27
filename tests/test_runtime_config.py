from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from memory.config import DEFAULT_MODELS, discover_models, load_selection, save_selection


class SelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_file = Path(self.temp_dir.name) / "memory-config.json"
        patcher = mock.patch.dict(os.environ, {"MEMORY_CONFIG": str(self.config_file)})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp_dir.cleanup)
        for name in ("MEMORY_PROVIDER", "MODEL"):
            os.environ.pop(name, None)

    def test_defaults_when_nothing_is_configured(self) -> None:
        self.assertEqual(load_selection(), ("lmstudio", DEFAULT_MODELS["lmstudio"]))

    def test_environment_variables_are_used(self) -> None:
        os.environ["MEMORY_PROVIDER"] = "ollama"
        os.environ["MODEL"] = "custom:1b"

        self.assertEqual(load_selection(), ("ollama", "custom:1b"))

    def test_saved_selection_overrides_the_environment(self) -> None:
        os.environ["MEMORY_PROVIDER"] = "gemini"
        save_selection("ollama", "qwen2.5:1.6b")

        self.assertEqual(load_selection(), ("ollama", "qwen2.5:1.6b"))

    def test_save_writes_provider_and_model(self) -> None:
        save_selection("lmstudio", "liquid/lfm2.5-1.2b")

        payload = json.loads(self.config_file.read_text(encoding="utf-8"))
        self.assertEqual(payload, {"provider": "lmstudio", "model": "liquid/lfm2.5-1.2b"})
        self.assertEqual(load_selection(), ("lmstudio", "liquid/lfm2.5-1.2b"))

    def test_saved_provider_without_model_uses_its_default(self) -> None:
        self.config_file.write_text(json.dumps({"provider": "ollama"}), encoding="utf-8")

        self.assertEqual(load_selection(), ("ollama", DEFAULT_MODELS["ollama"]))

    def test_corrupt_config_falls_back_to_the_environment(self) -> None:
        self.config_file.write_text("{not json", encoding="utf-8")
        os.environ["MEMORY_PROVIDER"] = "gemini"

        self.assertEqual(load_selection(), ("gemini", DEFAULT_MODELS["gemini"]))

    def test_unknown_saved_provider_is_ignored(self) -> None:
        self.config_file.write_text(json.dumps({"provider": "nope", "model": "x"}), encoding="utf-8")

        self.assertEqual(load_selection(), ("lmstudio", DEFAULT_MODELS["lmstudio"]))


class _JsonHandler(BaseHTTPRequestHandler):
    payload: dict = {}

    def do_GET(self) -> None:  # noqa: N802
        body = json.dumps(self.payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


class DiscoverModelsTests(unittest.TestCase):
    def serve(self, payload: dict) -> int:
        handler = type("StubHandler", (_JsonHandler,), {"payload": payload})
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def test_lists_lm_studio_models_sorted_and_unique(self) -> None:
        port = self.serve({"data": [{"id": "b-model"}, {"id": "a-model"}, {"id": "a-model"}]})

        with mock.patch.dict(os.environ, {"LMSTUDIO_API_BASE": f"http://127.0.0.1:{port}/v1"}):
            self.assertEqual(discover_models("lmstudio"), ["a-model", "b-model"])

    def test_lists_ollama_tags(self) -> None:
        port = self.serve({"models": [{"name": "qwen2.5:1.6b"}]})

        with mock.patch.dict(os.environ, {"OLLAMA_API_BASE": f"http://127.0.0.1:{port}"}):
            self.assertEqual(discover_models("ollama"), ["qwen2.5:1.6b"])

    def test_returns_empty_when_the_local_server_is_unreachable(self) -> None:
        with mock.patch.dict(os.environ, {"LMSTUDIO_API_BASE": "http://127.0.0.1:9/v1"}):
            self.assertEqual(discover_models("lmstudio", timeout=0.5), [])

    def test_gemini_has_no_local_discovery(self) -> None:
        self.assertEqual(discover_models("gemini"), [])


if __name__ == "__main__":
    unittest.main()
