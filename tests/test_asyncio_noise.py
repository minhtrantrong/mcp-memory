from __future__ import annotations

import logging
import sys
import unittest

from asyncio_noise import DropProactorConnectionReset, quiet_windows_asyncio_resets


def _record(message: str, exception: BaseException | None = None) -> logging.LogRecord:
    exc_info = (type(exception), exception, None) if exception is not None else None
    return logging.LogRecord("asyncio", logging.ERROR, __file__, 1, message, None, exc_info)


class DropProactorConnectionResetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.filter = DropProactorConnectionReset()

    def test_drops_connection_lost_reset_record(self) -> None:
        record = _record(
            "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
            ConnectionResetError(10054, "An existing connection was forcibly closed"),
        )

        self.assertFalse(self.filter.filter(record))

    def test_keeps_reset_records_from_other_callbacks(self) -> None:
        record = _record("Exception in callback something_else(None)", ConnectionResetError(10054, "reset"))

        self.assertTrue(self.filter.filter(record))

    def test_keeps_other_asyncio_errors(self) -> None:
        record = _record(
            "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)",
            ValueError("unexpected"),
        )

        self.assertTrue(self.filter.filter(record))

    def test_keeps_records_without_exception_info(self) -> None:
        self.assertTrue(self.filter.filter(_record("Some asyncio message")))

    @unittest.skipUnless(sys.platform == "win32", "Windows-only workaround")
    def test_installs_once_on_the_asyncio_logger(self) -> None:
        logger = logging.getLogger("asyncio")
        quiet_windows_asyncio_resets()
        quiet_windows_asyncio_resets()

        installed = [item for item in logger.filters if isinstance(item, DropProactorConnectionReset)]
        try:
            self.assertEqual(len(installed), 1)
        finally:
            for item in installed:
                logger.removeFilter(item)


if __name__ == "__main__":
    unittest.main()
