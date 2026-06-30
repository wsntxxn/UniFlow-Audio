import unittest
from pathlib import Path

from utils.logging import LoggingLogger


class TestLoggingLogger(unittest.TestCase):
    def setUp(self):
        self.tmp_log_path = Path("./tmp_logging.txt")

    def test_logging_info(self):
        logger = LoggingLogger(filename=self.tmp_log_path,
                               level="INFO").create_instance()
        logger.info("logging information")
        self.assertTrue(self.tmp_log_path.exists())

    def tearDown(self):
        if self.tmp_log_path.exists():
            self.tmp_log_path.unlink()
