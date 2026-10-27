"""Silence a benign Windows asyncio Proactor teardown error.

CPython's Windows Proactor transport calls ``socket.shutdown()`` from inside a
``finally`` block while tearing a connection down. When the peer already reset
the connection — a browser tab closing, an MCP client reconnecting — that call
raises ``ConnectionResetError`` (WinError 10054) and asyncio logs

    Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)

for a connection that is already gone. Nothing in the application can act on it,
so drop just that record and keep every other asyncio error visible.
"""

from __future__ import annotations

import logging
import sys


class DropProactorConnectionReset(logging.Filter):
    """Drop asyncio teardown records caused by a peer resetting the socket."""

    def filter(self, record: logging.LogRecord) -> bool:
        exception = record.exc_info[1] if record.exc_info else None
        if not isinstance(exception, ConnectionResetError):
            return True
        return "connection_lost" not in record.getMessage()


def quiet_windows_asyncio_resets() -> None:
    """Install the filter once; other platforms never produce this noise."""
    if sys.platform != "win32":
        return
    logger = logging.getLogger("asyncio")
    if not any(isinstance(item, DropProactorConnectionReset) for item in logger.filters):
        logger.addFilter(DropProactorConnectionReset())
