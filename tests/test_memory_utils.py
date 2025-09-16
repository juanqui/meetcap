"""Additional tests for memory utility functions to improve coverage"""

from meetcap.utils.memory import (
    estimate_model_memory,
    get_fallback_model,
)


class TestMemoryUtilityFunctions:
    """Test standalone utility functions in memory module"""

    def test_estimate_model_memory_whisper_models(self):
        """Test memory estimation for whisper models"""
        # Test exact matches
        assert estimate_model_memory("stt", "whisper-large-v3") == 1500
        assert estimate_model_memory("stt", "whisper-small") == 500
        assert estimate_model_memory("stt", "whisper-tiny") == 100

        # Test partial matches
        assert estimate_model_memory("stt", "large-v3") == 1500
        assert estimate_model_memory("stt", "tiny") == 100

    def test_estimate_model_memory_mlx_models(self):
        """Test memory estimation for MLX models"""
        assert estimate_model_memory("stt", "mlx-whisper-large-v3-turbo") == 1500
        assert estimate_model_memory("stt", "mlx-community/whisper-large-v3-turbo") == 1500

    def test_estimate_model_memory_vosk_models(self):
        """Test memory estimation for Vosk models"""
        assert estimate_model_memory("stt", "vosk-small") == 500
        assert estimate_model_memory("stt", "vosk-standard") == 1800

    def test_estimate_model_memory_llm_models(self):
        """Test memory estimation for LLM models"""
        assert estimate_model_memory("llm", "qwen2.5-3b") == 3000
        assert estimate_model_memory("llm", "qwen2.5-7b") == 7000
        assert estimate_model_memory("llm", "qwen2.5-14b") == 14000

    def test_estimate_model_memory_defaults(self):
        """Test default memory estimates for unknown models"""
        # Unknown STT model should default to 1500MB
        assert estimate_model_memory("stt", "unknown-model") == 1500

        # Unknown LLM model should default to 4000MB
        assert estimate_model_memory("llm", "unknown-model") == 4000

    def test_get_fallback_model_whisper_fallbacks(self):
        """Test fallback model selection for whisper models"""
        # Test exact matches with high memory requirement
        assert get_fallback_model("whisper-large-v3", 1000) == "whisper-small"
        assert get_fallback_model("whisper-large-v3-turbo", 1000) == "whisper-small"
        assert get_fallback_model("mlx-community/whisper-large-v3-turbo", 1000) == "whisper-small"

        # Test with sufficient memory (should return None)
        assert get_fallback_model("whisper-large-v3", 2000) is None

    def test_get_fallback_model_vosk_fallbacks(self):
        """Test fallback model selection for Vosk models"""
        assert get_fallback_model("vosk-standard", 1000) == "vosk-small"
        assert get_fallback_model("vosk-standard", 2000) is None  # sufficient memory

    def test_get_fallback_model_llm_fallbacks(self):
        """Test fallback model selection for LLM models"""
        assert get_fallback_model("qwen2.5-7b", 5000) == "qwen2.5-3b"
        assert get_fallback_model("qwen2.5-14b", 10000) == "qwen2.5-7b"

        # Test with sufficient memory
        assert get_fallback_model("qwen2.5-7b", 8000) is None

    def test_get_fallback_model_no_fallback(self):
        """Test models with no fallback available"""
        assert get_fallback_model("unknown-model", 1000) is None
        assert get_fallback_model("whisper-tiny", 50) is None  # already smallest
