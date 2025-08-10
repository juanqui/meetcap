"""Mock implementations of ML models for testing"""

from unittest.mock import Mock


class MockWhisperModel:
    """Mock WhisperModel for testing"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.transcribe = Mock(return_value=([], Mock(language="en")))


class MockLlama:
    """Mock Llama model for testing"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.create_chat_completion = Mock(
            return_value={"choices": [{"message": {"content": "## summary\n\nTest summary"}}]}
        )


# Create mock modules
mock_faster_whisper = Mock()
mock_faster_whisper.WhisperModel = MockWhisperModel
mock_faster_whisper.__spec__ = Mock()

mock_llama_cpp = Mock()
mock_llama_cpp.Llama = MockLlama
mock_llama_cpp.__spec__ = Mock()
