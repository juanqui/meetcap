"""tests for Vosk transcription service"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from meetcap.services.transcription import (
    TranscriptResult,
    VoskTranscriptionService,
)

# skip all tests if vosk is not installed
try:
    import importlib.util

    VOSK_AVAILABLE = (
        importlib.util.find_spec("vosk") is not None
        and importlib.util.find_spec("soundfile") is not None
    )
except ImportError:
    VOSK_AVAILABLE = False


@pytest.mark.skipif(not VOSK_AVAILABLE, reason="vosk not installed")
class TestVoskTranscriptionService:
    """test VoskTranscriptionService functionality"""

    @pytest.fixture
    def temp_model_dir(self, temp_dir):
        """create temporary model directory"""
        model_dir = temp_dir / "vosk-model"
        model_dir.mkdir()
        # create minimal structure for tests
        (model_dir / "conf").mkdir()
        (model_dir / "conf" / "model.conf").touch()
        return model_dir

    @pytest.fixture
    def temp_spk_model_dir(self, temp_dir):
        """create temporary speaker model directory"""
        spk_dir = temp_dir / "vosk-spk"
        spk_dir.mkdir()
        (spk_dir / "model").mkdir()
        (spk_dir / "model" / "final.raw").touch()
        return spk_dir

    def test_init(self, temp_model_dir):
        """test service initialization"""
        service = VoskTranscriptionService(
            model_path=str(temp_model_dir),
            spk_model_path=None,
            sample_rate=48000,
            enable_diarization=False,
        )

        assert service.model_path == temp_model_dir
        assert service.spk_model_path is None
        assert service.sample_rate == 48000
        assert service.enable_diarization is False
        assert service.model is None
        assert service.spk_model is None

    def test_init_with_speaker_model(self, temp_model_dir, temp_spk_model_dir):
        """test service initialization with speaker model"""
        service = VoskTranscriptionService(
            model_path=str(temp_model_dir),
            spk_model_path=str(temp_spk_model_dir),
            sample_rate=48000,
            enable_diarization=True,
        )

        assert service.model_path == temp_model_dir
        assert service.spk_model_path == temp_spk_model_dir
        assert service.enable_diarization is True

    def test_load_model(self, temp_model_dir):
        """test model loading"""
        with patch("vosk.SetLogLevel") as mock_set_log, patch("vosk.Model") as mock_model_class:
            mock_model = Mock()
            mock_model_class.return_value = mock_model

            service = VoskTranscriptionService(
                model_path=str(temp_model_dir),
                enable_diarization=False,
            )

            service._load_model()

            mock_set_log.assert_called_once_with(-1)
            mock_model_class.assert_called_once_with(str(temp_model_dir))
            assert service.model == mock_model

    def test_load_model_with_speaker(self, temp_model_dir, temp_spk_model_dir):
        """test model loading with speaker model"""
        with (
            patch("vosk.SetLogLevel"),
            patch("vosk.Model") as mock_model_class,
            patch("vosk.SpkModel") as mock_spk_class,
        ):
            mock_model = Mock()
            mock_spk_model = Mock()
            mock_model_class.return_value = mock_model
            mock_spk_class.return_value = mock_spk_model

            service = VoskTranscriptionService(
                model_path=str(temp_model_dir),
                spk_model_path=str(temp_spk_model_dir),
                enable_diarization=True,
            )

            service._load_model()

            mock_model_class.assert_called_once_with(str(temp_model_dir))
            mock_spk_class.assert_called_once_with(str(temp_spk_model_dir))
            assert service.model == mock_model
            assert service.spk_model == mock_spk_model

    def test_load_model_not_found(self):
        """test error when model not found"""
        service = VoskTranscriptionService(
            model_path="/nonexistent/model",
            enable_diarization=False,
        )

        with pytest.raises(FileNotFoundError):
            service._load_model()

    def test_transcribe_basic(self, temp_model_dir, temp_dir):
        """test basic transcription"""
        # create test audio file
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        with (
            patch("soundfile.read") as mock_sf_read,
            patch("vosk.SetLogLevel"),
            patch("vosk.Model") as mock_model_class,
            patch("vosk.KaldiRecognizer") as mock_recognizer_class,
        ):
            # mock soundfile
            import numpy as np

            mock_audio_data = np.random.rand(48000)  # 1 second of audio
            mock_sf_read.return_value = (mock_audio_data, 48000)

            # mock vosk
            mock_model = Mock()
            mock_recognizer = Mock()
            mock_model_class.return_value = mock_model
            mock_recognizer_class.return_value = mock_recognizer

            # mock recognition results
            mock_recognizer.AcceptWaveform.return_value = True  # simulate chunk completion
            mock_recognizer.Result.return_value = json.dumps(
                {
                    "result": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "conf": 0.95},
                        {"word": "world", "start": 0.6, "end": 1.0, "conf": 0.90},
                    ]
                }
            )
            mock_recognizer.FinalResult.return_value = json.dumps({"result": []})

            service = VoskTranscriptionService(
                model_path=str(temp_model_dir),
                enable_diarization=False,
            )

            result = service.transcribe(audio_file)

            assert isinstance(result, TranscriptResult)
            assert result.audio_path == str(audio_file)
            assert result.sample_rate == 48000
            assert result.language == "en"
            assert len(result.segments) > 0
            assert result.stt["engine"] == "vosk"
            assert result.diarization_enabled is False

    def test_transcribe_with_diarization(self, temp_model_dir, temp_spk_model_dir, temp_dir):
        """test transcription with speaker diarization"""
        # create test audio file
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        with (
            patch("soundfile.read") as mock_sf_read,
            patch("vosk.SetLogLevel"),
            patch("vosk.Model") as mock_model_class,
            patch("vosk.SpkModel") as mock_spk_class,
            patch("vosk.KaldiRecognizer") as mock_recognizer_class,
        ):
            # mock soundfile
            import numpy as np

            mock_audio_data = np.random.rand(48000)  # 1 second of audio
            mock_sf_read.return_value = (mock_audio_data, 48000)

            # mock vosk
            mock_model = Mock()
            mock_spk_model = Mock()
            mock_recognizer = Mock()
            mock_model_class.return_value = mock_model
            mock_spk_class.return_value = mock_spk_model
            mock_recognizer_class.return_value = mock_recognizer

            # mock recognition results with speaker info
            mock_recognizer.AcceptWaveform.return_value = True  # simulate chunk completion
            mock_recognizer.Result.return_value = json.dumps(
                {
                    "result": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "conf": 0.95},
                        {"word": "world", "start": 0.6, "end": 1.0, "conf": 0.90},
                    ],
                    "spk": [0.1, 0.2, 0.3],  # mock speaker embedding
                }
            )
            mock_recognizer.FinalResult.return_value = json.dumps({"result": []})

            service = VoskTranscriptionService(
                model_path=str(temp_model_dir),
                spk_model_path=str(temp_spk_model_dir),
                enable_diarization=True,
            )

            result = service.transcribe(audio_file)

            assert isinstance(result, TranscriptResult)
            assert result.diarization_enabled is True
            assert result.speakers is not None
            mock_recognizer.SetSpkModel.assert_called_once_with(mock_spk_model)

    @patch("meetcap.services.transcription.FasterWhisperService")
    def test_fallback_to_whisper(self, mock_whisper_class, temp_dir, temp_model_dir):
        """test fallback to Whisper when Vosk fails"""
        # create test audio file
        audio_file = temp_dir / "test.wav"
        audio_file.touch()

        # mock whisper service
        mock_whisper = Mock()
        mock_whisper_class.return_value = mock_whisper
        mock_whisper.transcribe.return_value = TranscriptResult(
            audio_path=str(audio_file),
            sample_rate=48000,
            language="en",
            segments=[],
            duration=0.0,
            stt={"engine": "faster-whisper"},
        )

        # service with valid model path
        service = VoskTranscriptionService(
            model_path=str(temp_model_dir),
            enable_diarization=False,
        )

        # mock loading to succeed but then fail on transcribe
        with (
            patch("vosk.SetLogLevel"),
            patch("vosk.Model") as mock_model_class,
            patch("soundfile.read", side_effect=RuntimeError("soundfile failed")),
        ):
            mock_model_class.return_value = Mock()

            # should fallback to whisper
            result = service.transcribe(audio_file)

            mock_whisper_class.assert_called_once()
            mock_whisper.transcribe.assert_called_once_with(audio_file)
            assert result.stt["engine"] == "faster-whisper"

    def test_transcribe_nonexistent_file(self, temp_model_dir):
        """test error when audio file doesn't exist"""
        service = VoskTranscriptionService(
            model_path=str(temp_model_dir),
            enable_diarization=False,
        )

        with pytest.raises(FileNotFoundError):
            service.transcribe(Path("/nonexistent/audio.wav"))

    def test_transcribe_m4a_conversion(self, temp_model_dir, temp_dir):
        """test automatic conversion of M4A files to WAV"""
        # create test M4A file
        audio_file = temp_dir / "test.m4a"
        audio_file.touch()

        with (
            patch("subprocess.run") as mock_run,
            patch("soundfile.read") as mock_sf_read,
            patch("vosk.SetLogLevel"),
            patch("vosk.Model") as mock_model_class,
            patch("vosk.KaldiRecognizer") as mock_recognizer_class,
            patch("tempfile.NamedTemporaryFile") as mock_temp_file,
        ):
            # mock ffmpeg conversion
            mock_run.return_value.returncode = 0

            # mock temp file creation
            temp_wav = temp_dir / "temp.wav"
            temp_wav.touch()
            mock_temp_file.return_value.__enter__.return_value.name = str(temp_wav)

            # mock soundfile
            import numpy as np

            mock_audio_data = np.random.rand(48000)  # 1 second of audio
            mock_sf_read.return_value = (mock_audio_data, 48000)

            # mock vosk
            mock_model = Mock()
            mock_recognizer = Mock()
            mock_model_class.return_value = mock_model
            mock_recognizer_class.return_value = mock_recognizer

            # mock recognition results
            mock_recognizer.AcceptWaveform.return_value = True  # simulate chunk completion
            mock_recognizer.Result.return_value = json.dumps(
                {
                    "result": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "conf": 0.95},
                    ]
                }
            )
            mock_recognizer.FinalResult.return_value = json.dumps({"result": []})

            service = VoskTranscriptionService(
                model_path=str(temp_model_dir),
                enable_diarization=False,
            )

            result = service.transcribe(audio_file)

            # verify ffmpeg was called for conversion
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "ffmpeg" in call_args
            assert str(audio_file) in call_args
            assert "-f" in call_args
            assert "wav" in call_args

            # verify result
            assert isinstance(result, TranscriptResult)
            assert result.stt["engine"] == "vosk"


class TestVoskModelDownload:
    """test Vosk model download functions"""

    @patch("meetcap.services.model_download.urllib.request.urlopen")
    @patch("meetcap.services.model_download.zipfile.ZipFile")
    def test_ensure_vosk_model(self, mock_zipfile, mock_urlopen, temp_dir):
        """test vosk model download"""
        from meetcap.services.model_download import ensure_vosk_model

        # mock URL response
        mock_response = Mock()
        mock_response.headers.get.return_value = "1000000"  # 1MB
        mock_response.read.side_effect = [b"data" * 100, b""]  # simulate file data
        mock_urlopen.return_value.__enter__.return_value = mock_response

        # mock zip extraction
        mock_zip = Mock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        # ensure model doesn't exist initially
        model_dir = temp_dir / "vosk-model-small-en-us-0.15"
        assert not model_dir.exists()

        # mock zip extraction to create the model structure
        def mock_extract(path):
            model_dir.mkdir(parents=True)
            (model_dir / "conf").mkdir()
            (model_dir / "conf" / "model.conf").touch()

        mock_zip.extractall.side_effect = mock_extract

        result = ensure_vosk_model(
            model_name="vosk-model-small-en-us-0.15",
            models_dir=temp_dir,
        )

        assert result == model_dir
        mock_urlopen.assert_called_once()
        mock_zip.extractall.assert_called_once()

    def test_verify_vosk_model(self, temp_dir):
        """test vosk model verification"""
        from meetcap.services.model_download import verify_vosk_model

        # create model structure
        model_dir = temp_dir / "vosk-model-en-us-0.22"
        model_dir.mkdir()
        (model_dir / "conf").mkdir()
        (model_dir / "conf" / "model.conf").touch()
        (model_dir / "am").mkdir()
        (model_dir / "am" / "final.mdl").touch()

        # should pass verification
        with patch("importlib.util.find_spec", return_value=Mock()):
            result = verify_vosk_model(
                model_name="vosk-model-en-us-0.22",
                models_dir=temp_dir,
            )
            assert result is True

    def test_verify_vosk_model_missing_files(self, temp_dir):
        """test vosk model verification with missing files"""
        from meetcap.services.model_download import verify_vosk_model

        # create incomplete model structure
        model_dir = temp_dir / "vosk-model-en-us-0.22"
        model_dir.mkdir()
        (model_dir / "conf").mkdir()
        # missing model.conf

        with patch("importlib.util.find_spec", return_value=Mock()):
            result = verify_vosk_model(
                model_name="vosk-model-en-us-0.22",
                models_dir=temp_dir,
            )
            assert result is False

    def test_verify_vosk_model_not_installed(self, temp_dir):
        """test vosk model verification when vosk not installed"""
        from meetcap.services.model_download import verify_vosk_model

        # vosk not installed
        with patch("importlib.util.find_spec", return_value=None):
            result = verify_vosk_model(
                model_name="vosk-model-en-us-0.22",
                models_dir=temp_dir,
            )
            assert result is False
