"""Tests for cruijff_kit.utils.logger."""

import logging
import tempfile
from pathlib import Path

import pytest

from cruijff_kit.utils.logger import setup_logger, get_logger


@pytest.fixture(autouse=True)
def cleanup_loggers():
    """Remove test loggers after each test to prevent handler accumulation."""
    yield
    for name in list(logging.Logger.manager.loggerDict.keys()):
        if name.startswith("test_logger_"):
            logger = logging.getLogger(name)
            logger.handlers.clear()


class TestSetupLogger:
    """Tests for setup_logger."""

    def test_returns_logger(self):
        logger = setup_logger("test_logger_basic")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self):
        logger = setup_logger("test_logger_name")
        assert logger.name == "test_logger_name"

    def test_default_level_is_info(self):
        logger = setup_logger("test_logger_level")
        assert logger.level == logging.INFO

    def test_custom_level(self):
        logger = setup_logger("test_logger_debug", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_console_handler_added(self):
        logger = setup_logger("test_logger_console")
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_no_console_handler_when_disabled(self):
        logger = setup_logger("test_logger_noconsole", console=False)
        assert len(logger.handlers) == 0

    def test_no_duplicate_handlers_on_repeat_call(self):
        """Calling setup_logger twice should not add duplicate handlers."""
        logger1 = setup_logger("test_logger_dedup")
        n_handlers = len(logger1.handlers)
        logger2 = setup_logger("test_logger_dedup")
        assert logger1 is logger2
        assert len(logger2.handlers) == n_handlers

    def test_propagation_disabled(self):
        logger = setup_logger("test_logger_propagate")
        assert logger.propagate is False

    def test_file_handler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "test.log"
            logger = setup_logger(
                "test_logger_file", log_file=str(log_file), console=False
            )
            logger.info("hello from test")

            assert log_file.exists()
            contents = log_file.read_text()
            assert "hello from test" in contents

    def test_file_handler_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "subdir" / "deep" / "test.log"
            logger = setup_logger(
                "test_logger_mkdir", log_file=str(log_file), console=False
            )
            logger.info("nested")
            assert log_file.exists()

    def test_custom_format_string(self):
        """Custom format_string should apply to the console handler."""
        logger = setup_logger(
            "test_logger_fmt",
            console=True,
            format_string="CUSTOM_PREFIX: %(message)s",
        )
        # The console handler should use our custom format
        assert len(logger.handlers) == 1
        handler = logger.handlers[0]
        assert "CUSTOM_PREFIX" in handler.formatter._fmt

    def test_both_console_and_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "both.log"
            logger = setup_logger(
                "test_logger_both", log_file=str(log_file), console=True
            )
            assert len(logger.handlers) == 2


class TestGetLogger:
    """Tests for get_logger."""

    def test_returns_logger(self):
        logger = get_logger("test_logger_get")
        assert isinstance(logger, logging.Logger)

    def test_returns_same_instance(self):
        setup_logger("test_logger_same")
        logger = get_logger("test_logger_same")
        assert logger.name == "test_logger_same"
