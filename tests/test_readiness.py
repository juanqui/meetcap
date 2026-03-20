"""tests for meetcap TUI readiness checks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestReadinessResult:
    """test ReadinessResult dataclass."""

    def test_empty_is_ready(self) -> None:
        from meetcap.tui.readiness import ReadinessResult

        result = ReadinessResult()
        assert result.ready is True
        assert result.errors == []
        assert result.warnings == []
        assert result.summary() == "All systems ready"

    def test_with_error_is_not_ready(self) -> None:
        from meetcap.tui.readiness import ReadinessIssue, ReadinessResult

        result = ReadinessResult(
            issues=[
                ReadinessIssue(
                    component="stt",
                    severity="error",
                    message="parakeet-mlx not installed",
                    fix_hint="pip install parakeet-mlx",
                )
            ]
        )
        assert result.ready is False
        assert len(result.errors) == 1
        assert "1 error" in result.summary()

    def test_with_only_warnings_is_ready(self) -> None:
        from meetcap.tui.readiness import ReadinessIssue, ReadinessResult

        result = ReadinessResult(
            issues=[
                ReadinessIssue(
                    component="llm",
                    severity="warning",
                    message="LLM model not downloaded",
                    fix_hint="meetcap setup",
                )
            ]
        )
        assert result.ready is True
        assert len(result.warnings) == 1
        assert "1 warning" in result.summary()


class TestIsPackageInstalled:
    """test _is_package_installed helper."""

    def test_installed_package(self) -> None:
        from meetcap.tui.readiness import _is_package_installed

        assert _is_package_installed("os") is True
        assert _is_package_installed("pathlib") is True

    def test_missing_package(self) -> None:
        from meetcap.tui.readiness import _is_package_installed

        assert _is_package_installed("nonexistent_package_xyz") is False


class TestCheckFfmpeg:
    """test _check_ffmpeg."""

    def test_ffmpeg_found(self) -> None:
        from meetcap.tui.readiness import ReadinessResult, _check_ffmpeg

        result = ReadinessResult()
        _check_ffmpeg(result)
        # ffmpeg should be installed on this machine
        assert not any(i.component == "ffmpeg" for i in result.issues)

    def test_ffmpeg_not_found(self) -> None:
        from meetcap.tui.readiness import ReadinessResult, _check_ffmpeg

        result = ReadinessResult()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            _check_ffmpeg(result)
        ffmpeg_issues = [i for i in result.issues if i.component == "ffmpeg"]
        assert len(ffmpeg_issues) == 1
        assert ffmpeg_issues[0].severity == "error"


class TestCheckStt:
    """test _check_stt for various engines."""

    def test_parakeet_not_installed(self) -> None:
        from meetcap.tui.readiness import ReadinessResult, _check_stt

        result = ReadinessResult()
        mock_config = MagicMock()
        mock_config.expand_path.return_value = Path("/tmp/models")
        mock_config.get.return_value = "~/.meetcap/models"

        with patch("meetcap.tui.readiness._is_package_installed", return_value=False):
            _check_stt(result, mock_config, "parakeet")

        stt_issues = [i for i in result.issues if i.component == "stt"]
        assert len(stt_issues) == 1
        assert "parakeet-mlx" in stt_issues[0].message
        assert stt_issues[0].severity == "error"

    def test_parakeet_installed(self) -> None:
        from meetcap.tui.readiness import ReadinessResult, _check_stt

        result = ReadinessResult()
        mock_config = MagicMock()
        mock_config.expand_path.return_value = Path("/tmp/models")
        mock_config.get.return_value = "~/.meetcap/models"

        with patch("meetcap.tui.readiness._is_package_installed", return_value=True):
            _check_stt(result, mock_config, "parakeet")

        stt_issues = [i for i in result.issues if i.component == "stt"]
        assert len(stt_issues) == 0

    def test_fwhisper_not_installed(self) -> None:
        from meetcap.tui.readiness import ReadinessResult, _check_stt

        result = ReadinessResult()
        mock_config = MagicMock()
        mock_config.expand_path.return_value = Path("/tmp/models")
        mock_config.get.return_value = "~/.meetcap/models"

        with patch("meetcap.tui.readiness._is_package_installed", return_value=False):
            _check_stt(result, mock_config, "faster-whisper")

        stt_issues = [i for i in result.issues if i.component == "stt"]
        assert len(stt_issues) == 1
        assert "faster-whisper" in stt_issues[0].message

    def test_mlx_whisper_not_installed(self) -> None:
        from meetcap.tui.readiness import ReadinessResult, _check_stt

        result = ReadinessResult()
        mock_config = MagicMock()
        mock_config.expand_path.return_value = Path("/tmp/models")
        mock_config.get.return_value = "~/.meetcap/models"

        with patch("meetcap.tui.readiness._is_package_installed", return_value=False):
            _check_stt(result, mock_config, "mlx-whisper")

        stt_issues = [i for i in result.issues if i.component == "stt"]
        assert len(stt_issues) == 1
        assert "mlx-whisper" in stt_issues[0].message


class TestCheckReadiness:
    """test the full check_readiness function."""

    def test_returns_readiness_result(self) -> None:
        from meetcap.tui.readiness import ReadinessResult, check_readiness

        with patch("meetcap.utils.config.Config") as MockConfig:
            mock_config = MagicMock()
            mock_config.get.side_effect = lambda s, k, d=None: {
                ("models", "stt_engine"): "parakeet",
                ("models", "enable_speaker_diarization"): False,
                ("models", "diarization_backend"): "sherpa",
                ("models", "llm_model_name"): "mlx-community/Qwen3.5-4B-MLX-4bit",
            }.get((s, k), d)
            mock_config.expand_path.return_value = Path("/tmp/models")
            MockConfig.return_value = mock_config

            with patch("meetcap.tui.readiness._check_ffmpeg"):
                with patch("meetcap.tui.readiness._check_stt"):
                    with patch("meetcap.tui.readiness._check_llm"):
                        result = check_readiness()
                        assert isinstance(result, ReadinessResult)
