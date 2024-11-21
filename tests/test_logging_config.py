import pytest
import logging
import os
from unittest.mock import patch, MagicMock
from src.logging_config import setup_logging


@pytest.fixture
def mock_os():
    with patch("os.makedirs") as makedirs_mock, patch(
        "os.path.exists", return_value=False
    ) as exists_mock:
        yield makedirs_mock, exists_mock


def test_setup_logging_creates_logs_directory(mock_os):
    makedirs_mock, exists_mock = mock_os
    setup_logging()
    makedirs_mock.assert_called_once_with("logs")
    exists_mock.assert_called_once_with("logs")


def test_setup_logging_configures_logger():
    with patch("logging.FileHandler") as FileHandlerMock:
        # Mock FileHandlers for debug and error logs
        mock_debug_handler = MagicMock()
        mock_debug_handler.level = logging.DEBUG
        mock_error_handler = MagicMock()
        mock_error_handler.level = logging.ERROR
        FileHandlerMock.side_effect = [mock_debug_handler, mock_error_handler]

        # Call the setup_logging function
        logger = setup_logging()

        # Assertions for logger configuration
        assert logger.name == "user_actions"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) == 2

        # Extract handlers
        debug_handler = logger.handlers[0]
        error_handler = logger.handlers[1]

        # Validate handler levels
        assert debug_handler.level == logging.DEBUG
        assert error_handler.level == logging.ERROR

        # Validate formatter setting
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Check if setFormatter was called with any instance of Formatter
        debug_handler.setFormatter.assert_called_once()
        error_handler.setFormatter.assert_called_once()

        # Validate formatter properties explicitly
        actual_debug_formatter = debug_handler.setFormatter.call_args[0][0]
        actual_error_formatter = error_handler.setFormatter.call_args[0][0]

        assert actual_debug_formatter._fmt == formatter._fmt
        assert actual_error_formatter._fmt == formatter._fmt


def test_setup_logging_creates_log_files():
    with patch("logging.FileHandler") as FileHandlerMock:
        setup_logging()
        FileHandlerMock.assert_any_call("logs/debug.log")
        FileHandlerMock.assert_any_call("logs/error.log")
