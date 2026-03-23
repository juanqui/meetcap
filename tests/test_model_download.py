"""tests for model download service"""

from pathlib import Path
from unittest.mock import Mock, patch

from meetcap.services.model_download import (
    ensure_mlx_llm_model,
    ensure_mlx_whisper_model,
    ensure_whisper_model,
    verify_mlx_llm_model,
    verify_mlx_whisper_model,
    verify_whisper_model,
)


class TestVerifyFunctions:
    """test model verification functions"""

    @patch("meetcap.services.model_download.console")
    def test_verify_whisper_model_no_import(self, mock_console):
        """test verifying whisper model without faster-whisper"""
        with patch("importlib.util.find_spec", return_value=None):
            result = verify_whisper_model("test-model")
            assert result is False

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_llm_model_not_arm(self, mock_console):
        """test verifying mlx llm model on non-ARM processor"""
        with patch("platform.processor", return_value="x86_64"):
            result = verify_mlx_llm_model("mlx-community/Qwen3.5-2B-OptiQ-4bit")
            assert result is False

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_llm_model_no_import(self, mock_console):
        """test verifying mlx llm model without mlx-vlm installed"""
        with patch("platform.processor", return_value="arm"):
            with patch("importlib.util.find_spec", return_value=None):
                result = verify_mlx_llm_model("mlx-community/Qwen3.5-2B-OptiQ-4bit")
                assert result is False

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_llm_model_cached(self, mock_console):
        """test verifying mlx llm model when cached"""
        with patch("platform.processor", return_value="arm"):
            with patch("importlib.util.find_spec", return_value=Mock()):
                with patch(
                    "huggingface_hub.try_to_load_from_cache",
                    return_value="/cached/config.json",
                ):
                    result = verify_mlx_llm_model("mlx-community/Qwen3.5-2B-OptiQ-4bit")
                    assert result is True

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_llm_model_not_cached(self, mock_console):
        """test verifying mlx llm model when not cached"""
        with patch("platform.processor", return_value="arm"):
            with patch("importlib.util.find_spec", return_value=Mock()):
                with patch("huggingface_hub.try_to_load_from_cache", return_value=None):
                    result = verify_mlx_llm_model("mlx-community/Qwen3.5-2B-OptiQ-4bit")
                    assert result is False

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_llm_model_error(self, mock_console):
        """test verifying mlx llm model with error"""
        with patch("platform.processor", return_value="arm"):
            with patch("importlib.util.find_spec", return_value=Mock()):
                with patch(
                    "huggingface_hub.try_to_load_from_cache",
                    side_effect=Exception("cache error"),
                ):
                    result = verify_mlx_llm_model("mlx-community/Qwen3.5-2B-OptiQ-4bit")
                    assert result is False


class TestEnsureFunctions:
    """test model ensure functions"""

    @patch("meetcap.services.model_download.Progress")
    @patch("meetcap.services.model_download.console")
    def test_ensure_whisper_model_invalid_name(self, mock_console, mock_progress, temp_dir):
        """test ensuring whisper model with invalid name"""
        result = ensure_whisper_model("invalid-model", temp_dir)
        assert result is None or isinstance(result, str | Path)

    @patch("meetcap.services.model_download.console")
    def test_ensure_mlx_llm_model_success(self, mock_console):
        """test successful mlx llm model download"""
        with patch("huggingface_hub.snapshot_download") as mock_download:
            result = ensure_mlx_llm_model("mlx-community/Qwen3.5-2B-OptiQ-4bit")
            assert result is True
            mock_download.assert_called_once_with("mlx-community/Qwen3.5-2B-OptiQ-4bit")

    @patch("meetcap.services.model_download.console")
    def test_ensure_mlx_llm_model_failure(self, mock_console):
        """test failed mlx llm model download"""
        with patch("huggingface_hub.snapshot_download", side_effect=Exception("download failed")):
            result = ensure_mlx_llm_model("mlx-community/Qwen3.5-2B-OptiQ-4bit")
            assert result is False

    @patch("meetcap.services.model_download.console")
    def test_ensure_mlx_llm_model_default_name(self, mock_console):
        """test ensure_mlx_llm_model uses default model name"""
        with patch("huggingface_hub.snapshot_download") as mock_download:
            result = ensure_mlx_llm_model()
            assert result is True
            mock_download.assert_called_once_with("mlx-community/Qwen3.5-2B-OptiQ-4bit")


class TestMlxWhisperFunctions:
    """test mlx-whisper model functions"""

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_whisper_model_no_import(self, mock_console):
        """test verifying mlx-whisper model without mlx-whisper installed"""
        with patch("importlib.util.find_spec", return_value=None):
            result = verify_mlx_whisper_model("mlx-community/whisper-large-v3-turbo")
            assert result is False

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_whisper_model_not_arm(self, mock_console):
        """test verifying mlx-whisper model on non-ARM processor"""
        with patch("importlib.util.find_spec", return_value=Mock()):
            with patch("platform.processor", return_value="x86_64"):
                result = verify_mlx_whisper_model("mlx-community/whisper-large-v3-turbo")
                assert result is False

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_whisper_model_success(self, mock_console):
        """test successful mlx-whisper model verification"""
        mock_mlx_whisper = Mock()

        with patch("importlib.util.find_spec", return_value=Mock()):
            with patch("platform.processor", return_value="arm"):
                with patch.dict("sys.modules", {"mlx_whisper": mock_mlx_whisper}):
                    result = verify_mlx_whisper_model("mlx-community/whisper-large-v3-turbo")
                    assert result is True
                    mock_mlx_whisper.transcribe.assert_called_once()

    @patch("meetcap.services.model_download.console")
    def test_verify_mlx_whisper_model_load_error(self, mock_console):
        """test mlx-whisper model verification with load error"""
        mock_mlx_whisper = Mock()
        mock_mlx_whisper.transcribe.side_effect = Exception("Model load failed")

        with patch("importlib.util.find_spec", return_value=Mock()):
            with patch("platform.processor", return_value="arm"):
                with patch.dict("sys.modules", {"mlx_whisper": mock_mlx_whisper}):
                    result = verify_mlx_whisper_model("mlx-community/whisper-large-v3-turbo")
                    assert result is False

    @patch("meetcap.services.model_download.Progress")
    @patch("meetcap.services.model_download.console")
    def test_ensure_mlx_whisper_model_no_import(self, mock_console, mock_progress, temp_dir):
        """test ensuring mlx-whisper model without mlx-whisper installed"""
        original_import = __builtins__["__import__"]

        def mock_import(name, *args, **kwargs):
            if name == "mlx_whisper":
                raise ImportError("No module named 'mlx_whisper'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = ensure_mlx_whisper_model("mlx-community/whisper-large-v3-turbo", temp_dir)
            assert result is None

    @patch("meetcap.services.model_download.Progress")
    @patch("meetcap.services.model_download.console")
    def test_ensure_mlx_whisper_model_exists(self, mock_console, mock_progress, temp_dir):
        """test ensuring existing mlx-whisper model"""
        model_dir = temp_dir / "mlx-community--whisper-large-v3-turbo"
        model_dir.mkdir(parents=True)

        result = ensure_mlx_whisper_model("mlx-community/whisper-large-v3-turbo", temp_dir)
        assert result == model_dir

    @patch("meetcap.services.model_download.Progress")
    @patch("meetcap.services.model_download.console")
    def test_ensure_mlx_whisper_model_download_success(self, mock_console, mock_progress, temp_dir):
        """test successful mlx-whisper model download"""
        mock_mlx_whisper = Mock()

        with patch.dict("sys.modules", {"mlx_whisper": mock_mlx_whisper}):
            result = ensure_mlx_whisper_model("mlx-community/whisper-large-v3-turbo", temp_dir)

            expected_path = temp_dir / "mlx-community--whisper-large-v3-turbo"
            assert result == expected_path
            mock_mlx_whisper.transcribe.assert_called_once()

    @patch("meetcap.services.model_download.Progress")
    @patch("meetcap.services.model_download.console")
    def test_ensure_mlx_whisper_model_download_error(self, mock_console, mock_progress, temp_dir):
        """test mlx-whisper model download error"""
        mock_mlx_whisper = Mock()
        mock_mlx_whisper.transcribe.side_effect = Exception("Download failed")

        with patch.dict("sys.modules", {"mlx_whisper": mock_mlx_whisper}):
            result = ensure_mlx_whisper_model("mlx-community/whisper-large-v3-turbo", temp_dir)
            assert result is None
