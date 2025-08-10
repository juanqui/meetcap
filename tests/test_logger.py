"""comprehensive tests for logging and error handling"""

import logging
from unittest.mock import patch

import pytest

from meetcap.utils.logger import ErrorHandler, MeetcapLogger, logger


class TestMeetcapLogger:
    """test meetcap logger functionality"""

    @pytest.fixture
    def fresh_logger(self):
        """create a fresh logger instance"""
        # reset singleton
        MeetcapLogger._instance = None
        MeetcapLogger._logger = None
        return MeetcapLogger()

    def test_singleton_pattern(self):
        """test logger singleton behavior"""
        logger1 = MeetcapLogger()
        logger2 = MeetcapLogger()

        assert logger1 is logger2
        assert logger1._logger is logger2._logger

    def test_initialization(self, fresh_logger):
        """test logger initialization"""
        assert fresh_logger._logger is not None
        assert fresh_logger._logger.name == "meetcap"
        assert fresh_logger._logger.level == logging.INFO

        # check handlers
        handlers = fresh_logger._logger.handlers
        assert len(handlers) >= 1

        # verify rich handler
        from rich.logging import RichHandler

        assert any(isinstance(h, RichHandler) for h in handlers)

    def test_add_file_handler(self, fresh_logger, temp_dir):
        """test adding file handler"""
        log_file = temp_dir / "test.log"

        fresh_logger.add_file_handler(log_file, level=logging.DEBUG)

        # verify file handler was added
        handlers = fresh_logger._logger.handlers
        file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

        # verify log file directory was created
        assert log_file.parent.exists()

        # test logging to file
        fresh_logger.info("test message")

        # file might not be flushed immediately, so force flush
        for handler in file_handlers:
            handler.flush()

        if log_file.exists():
            content = log_file.read_text()
            assert "test message" in content

    def test_add_file_handler_creates_directory(self, fresh_logger, temp_dir):
        """test file handler creates parent directory"""
        log_file = temp_dir / "subdir" / "nested" / "test.log"

        fresh_logger.add_file_handler(log_file)

        assert log_file.parent.exists()
        assert log_file.parent.is_dir()

    def test_set_level(self, fresh_logger):
        """test setting log level"""
        fresh_logger.set_level(logging.DEBUG)
        assert fresh_logger._logger.level == logging.DEBUG

        fresh_logger.set_level(logging.WARNING)
        assert fresh_logger._logger.level == logging.WARNING

    def test_debug_logging(self, fresh_logger):
        """test debug message logging"""
        with patch.object(fresh_logger._logger, "debug") as mock_debug:
            fresh_logger.debug("debug message")
            mock_debug.assert_called_once_with("debug message")

    def test_info_logging(self, fresh_logger):
        """test info message logging"""
        with patch.object(fresh_logger._logger, "info") as mock_info:
            fresh_logger.info("info message")
            mock_info.assert_called_once_with("info message")

    def test_warning_logging(self, fresh_logger):
        """test warning message logging"""
        with patch.object(fresh_logger._logger, "warning") as mock_warning:
            fresh_logger.warning("warning message")
            mock_warning.assert_called_once_with("warning message")

    def test_error_logging(self, fresh_logger):
        """test error message logging"""
        with patch.object(fresh_logger._logger, "error") as mock_error:
            fresh_logger.error("error message")
            mock_error.assert_called_once_with("error message", exc_info=False)

            fresh_logger.error("error with trace", exc_info=True)
            mock_error.assert_called_with("error with trace", exc_info=True)

    def test_critical_logging(self, fresh_logger):
        """test critical message logging"""
        with patch.object(fresh_logger._logger, "critical") as mock_critical:
            fresh_logger.critical("critical message")
            mock_critical.assert_called_once_with("critical message", exc_info=False)

            fresh_logger.critical("critical with trace", exc_info=True)
            mock_critical.assert_called_with("critical with trace", exc_info=True)

    def test_global_logger_instance(self):
        """test global logger instance"""
        from meetcap.utils.logger import logger

        assert isinstance(logger, MeetcapLogger)
        assert logger._logger is not None
        assert logger._logger.name == "meetcap"


class TestErrorHandler:
    """test error handler functionality"""

    def test_exit_codes(self):
        """test exit code constants"""
        assert ErrorHandler.EXIT_SUCCESS == 0
        assert ErrorHandler.EXIT_GENERAL_ERROR == 1
        assert ErrorHandler.EXIT_CONFIG_ERROR == 2
        assert ErrorHandler.EXIT_PERMISSION_ERROR == 3
        assert ErrorHandler.EXIT_RUNTIME_ERROR == 4

    def test_handle_config_error(self):
        """test configuration error handling"""
        error = Exception("invalid config value")

        with patch("meetcap.utils.logger.console") as mock_console:
            with patch("sys.exit") as mock_exit:
                ErrorHandler.handle_config_error(error)
                mock_exit.assert_called_once_with(ErrorHandler.EXIT_CONFIG_ERROR)

            # verify console output
            calls = mock_console.print.call_args_list
            output = " ".join(str(call) for call in calls).lower()

            assert "configuration error" in output
            assert "invalid config value" in output
            assert "suggestions" in output
            assert "config.toml" in output
            assert "meetcap verify" in output

    def test_handle_permission_error(self):
        """test permission error handling"""
        error = Exception("access denied")

        with patch("meetcap.utils.logger.console") as mock_console:
            with patch("sys.exit") as mock_exit:
                ErrorHandler.handle_permission_error(error)
                mock_exit.assert_called_once_with(ErrorHandler.EXIT_PERMISSION_ERROR)

            # verify console output
            calls = mock_console.print.call_args_list
            output = " ".join(str(call) for call in calls).lower()

            assert "permission error" in output
            assert "access denied" in output
            assert "microphone access" in output
            assert "input monitoring" in output
            assert "privacy & security" in output

    def test_handle_runtime_error_ffmpeg(self):
        """test runtime error handling for ffmpeg issues"""
        error = Exception("ffmpeg not found")

        with patch("meetcap.utils.logger.console") as mock_console:
            with patch("sys.exit") as mock_exit:
                ErrorHandler.handle_runtime_error(error)
                mock_exit.assert_called_once_with(ErrorHandler.EXIT_RUNTIME_ERROR)

            # verify ffmpeg-specific suggestions
            calls = mock_console.print.call_args_list
            output = " ".join(str(call) for call in calls).lower()

            assert "runtime error" in output
            assert "ffmpeg issue detected" in output
            assert "brew install ffmpeg" in output
            assert "meetcap devices" in output

    def test_handle_runtime_error_model(self):
        """test runtime error handling for model issues"""
        error = Exception("failed to load whisper model")

        with patch("meetcap.utils.logger.console") as mock_console:
            with patch("sys.exit") as mock_exit:
                ErrorHandler.handle_runtime_error(error)
                mock_exit.assert_called_once_with(ErrorHandler.EXIT_RUNTIME_ERROR)

            # verify model-specific suggestions
            calls = mock_console.print.call_args_list
            output = " ".join(str(call) for call in calls).lower()

            assert "model loading issue" in output
            assert "verify model files" in output
            assert "disk space" in output
            assert "gguf" in output

    def test_handle_runtime_error_llama(self):
        """test runtime error handling for llama issues"""
        error = Exception("llama.cpp error")

        with patch("meetcap.utils.logger.console") as mock_console:
            with patch("sys.exit") as mock_exit:
                ErrorHandler.handle_runtime_error(error)
                mock_exit.assert_called_once_with(ErrorHandler.EXIT_RUNTIME_ERROR)

            # verify llama triggers model suggestions
            calls = mock_console.print.call_args_list
            output = " ".join(str(call) for call in calls).lower()

            assert "model loading issue" in output

    def test_handle_runtime_error_disk(self):
        """test runtime error handling for disk space issues"""
        error = Exception("no space left on disk")

        with patch("meetcap.utils.logger.console") as mock_console:
            with patch("sys.exit") as mock_exit:
                ErrorHandler.handle_runtime_error(error)
                mock_exit.assert_called_once_with(ErrorHandler.EXIT_RUNTIME_ERROR)

            # verify disk-specific suggestions
            calls = mock_console.print.call_args_list
            output = " ".join(str(call) for call in calls).lower()

            assert "storage issue" in output
            assert "available disk space" in output
            assert "output directory" in output

    def test_handle_runtime_error_generic(self):
        """test runtime error handling for generic errors"""
        error = Exception("some other error")

        with patch("meetcap.utils.logger.console") as mock_console:
            with patch("sys.exit") as mock_exit:
                ErrorHandler.handle_runtime_error(error)
                mock_exit.assert_called_once_with(ErrorHandler.EXIT_RUNTIME_ERROR)

            # should still show the error
            calls = mock_console.print.call_args_list
            output = " ".join(str(call) for call in calls).lower()

            assert "runtime error" in output
            assert "some other error" in output

    def test_handle_general_error(self):
        """test general error handling"""
        error = Exception("unexpected problem")

        with patch("meetcap.utils.logger.console") as mock_console:
            with patch.object(logger, "error") as mock_logger_error:
                with patch("sys.exit") as mock_exit:
                    ErrorHandler.handle_general_error(error)
                    mock_exit.assert_called_once_with(ErrorHandler.EXIT_GENERAL_ERROR)

                # verify console output
                calls = mock_console.print.call_args_list
                output = " ".join(str(call) for call in calls).lower()

                assert "unexpected error" in output
                assert "unexpected problem" in output
                assert "report this issue" in output

                # verify logging
                mock_logger_error.assert_called_once_with("uncaught exception", exc_info=True)


class TestLoggerIntegration:
    """integration tests for logging system"""

    def test_file_logging_integration(self, temp_dir):
        """test complete file logging flow"""
        log_file = temp_dir / "integration.log"

        # create logger and add file handler
        test_logger = MeetcapLogger()
        test_logger.add_file_handler(log_file, level=logging.DEBUG)
        test_logger.set_level(logging.DEBUG)

        # log messages at different levels
        test_logger.debug("debug message")
        test_logger.info("info message")
        test_logger.warning("warning message")
        test_logger.error("error message")
        test_logger.critical("critical message")

        # flush handlers
        for handler in test_logger._logger.handlers:
            if hasattr(handler, "flush"):
                handler.flush()

        # verify file content if it exists
        if log_file.exists():
            content = log_file.read_text()
            assert "debug message" in content
            assert "info message" in content
            assert "warning message" in content
            assert "error message" in content
            assert "critical message" in content

            # verify format
            assert "DEBUG" in content
            assert "INFO" in content
            assert "WARNING" in content
            assert "ERROR" in content
            assert "CRITICAL" in content

    def test_error_handler_with_logger(self):
        """test error handler integration with logger"""
        error = Exception("test error")

        with patch("meetcap.utils.logger.console"):
            with patch.object(logger, "error") as mock_logger:
                with patch("sys.exit") as mock_exit:
                    ErrorHandler.handle_general_error(error)
                    mock_exit.assert_called_once_with(ErrorHandler.EXIT_GENERAL_ERROR)

            mock_logger.assert_called_once_with("uncaught exception", exc_info=True)

    def test_multiple_file_handlers(self, temp_dir):
        """test adding multiple file handlers"""
        log_file1 = temp_dir / "log1.log"
        log_file2 = temp_dir / "log2.log"

        test_logger = MeetcapLogger()
        test_logger.add_file_handler(log_file1, level=logging.INFO)
        test_logger.add_file_handler(log_file2, level=logging.DEBUG)

        # verify multiple handlers
        file_handlers = [
            h for h in test_logger._logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 2
