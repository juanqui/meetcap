"""comprehensive tests for configuration management"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import toml

from meetcap.utils.config import Config


class TestConfig:
    """test configuration management"""

    @pytest.fixture
    def config_file(self, temp_dir):
        """create a temporary config file path"""
        return temp_dir / "test_config.toml"

    @pytest.fixture
    def sample_config_data(self):
        """sample configuration data"""
        return {
            "audio": {
                "preferred_device": "Test Device",
                "sample_rate": 44100,
            },
            "hotkey": {
                "stop": "<ctrl>+q",
            },
            "paths": {
                "out_dir": "/custom/output",
            },
        }

    def test_init_default_path(self):
        """test initialization with default config path"""
        with patch("pathlib.Path.exists", return_value=False):
            config = Config()

            expected_path = Path.home() / ".meetcap" / "config.toml"
            assert config.config_path == expected_path
            assert config.config == Config.DEFAULT_CONFIG

    def test_init_custom_path(self, config_file):
        """test initialization with custom config path"""
        with patch("pathlib.Path.exists", return_value=False):
            config = Config(config_path=config_file)

            assert config.config_path == config_file
            assert config.config == Config.DEFAULT_CONFIG

    def test_load_from_file_success(self, config_file, sample_config_data):
        """test loading configuration from file"""
        # write config file
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            toml.dump(sample_config_data, f)

        config = Config(config_path=config_file)

        # verify loaded values
        assert config.get("audio", "preferred_device") == "Test Device"
        assert config.get("audio", "sample_rate") == 44100
        assert config.get("hotkey", "stop") == "<ctrl>+q"
        assert config.get("paths", "out_dir") == "/custom/output"

        # verify defaults are preserved for missing keys
        assert config.get("audio", "channels") == 2  # from default
        assert config.get("llm", "temperature") == 0.4  # from default

    def test_load_from_file_error(self, config_file):
        """test handling of file load errors"""
        config_file.write_text("invalid toml content{")

        with patch("meetcap.utils.config.console") as mock_console:
            config = Config(config_path=config_file)

            # should use defaults on error
            assert config.config == Config.DEFAULT_CONFIG

            # should print warning
            mock_console.print.assert_called()
            warning_call = str(mock_console.print.call_args)
            assert "warning" in warning_call.lower()
            assert "failed to load config" in warning_call.lower()

    def test_apply_env_overrides(self):
        """test environment variable overrides"""
        env_vars = {
            "MEETCAP_DEVICE": "Env Device",
            "MEETCAP_SAMPLE_RATE": "96000",
            "MEETCAP_CHANNELS": "1",
            "MEETCAP_HOTKEY": "<env>+<key>",
            "MEETCAP_STT_MODEL": "/env/stt/model",
            "MEETCAP_LLM_MODEL": "/env/llm/model",
            "MEETCAP_OUT_DIR": "/env/output",
            "MEETCAP_N_CTX": "4096",
            "MEETCAP_N_THREADS": "8",
            "MEETCAP_N_GPU_LAYERS": "20",
        }

        with patch.dict(os.environ, env_vars):
            with patch("pathlib.Path.exists", return_value=False):
                config = Config()

        # verify string overrides
        assert config.get("audio", "preferred_device") == "Env Device"
        assert config.get("hotkey", "stop") == "<env>+<key>"
        assert config.get("models", "stt_model_path") == "/env/stt/model"
        assert config.get("models", "llm_gguf_path") == "/env/llm/model"
        assert config.get("paths", "out_dir") == "/env/output"

        # verify integer overrides
        assert config.get("audio", "sample_rate") == 96000
        assert config.get("audio", "channels") == 1
        assert config.get("llm", "n_ctx") == 4096
        assert config.get("llm", "n_threads") == 8
        assert config.get("llm", "n_gpu_layers") == 20

    def test_env_override_invalid_int(self):
        """test handling of invalid integer env values"""
        with patch.dict(os.environ, {"MEETCAP_SAMPLE_RATE": "not_a_number"}):
            with patch("pathlib.Path.exists", return_value=False):
                config = Config()

        # should keep default on invalid conversion
        assert config.get("audio", "sample_rate") == 48000

    def test_deep_merge(self):
        """test deep merge functionality"""
        config = Config()

        base = {
            "section1": {
                "key1": "value1",
                "key2": "value2",
                "nested": {
                    "deep1": "original",
                },
            },
            "section2": {
                "key3": "value3",
            },
        }

        update = {
            "section1": {
                "key2": "updated",
                "key4": "new",
                "nested": {
                    "deep1": "updated",
                    "deep2": "new",
                },
            },
            "section3": {
                "key5": "value5",
            },
        }

        config._deep_merge(base, update)

        # verify merge results
        assert base["section1"]["key1"] == "value1"  # unchanged
        assert base["section1"]["key2"] == "updated"  # updated
        assert base["section1"]["key4"] == "new"  # added
        assert base["section1"]["nested"]["deep1"] == "updated"  # deep update
        assert base["section1"]["nested"]["deep2"] == "new"  # deep add
        assert base["section2"]["key3"] == "value3"  # unchanged
        assert base["section3"]["key5"] == "value5"  # new section

    def test_get(self):
        """test getting configuration values"""
        config = Config()

        # existing value
        assert config.get("audio", "sample_rate") == 48000

        # with default
        assert config.get("nonexistent", "key", "default") == "default"

        # nested value
        assert config.get("llm", "temperature") == 0.4

    def test_get_section(self):
        """test getting configuration sections"""
        config = Config()

        # existing section
        audio_section = config.get_section("audio")
        assert audio_section["sample_rate"] == 48000
        assert audio_section["channels"] == 2

        # nonexistent section
        assert config.get_section("nonexistent") == {}

    def test_save(self, config_file):
        """test saving configuration to file"""
        config = Config(config_path=config_file)
        config.config["test_key"] = "test_value"

        with patch("meetcap.utils.config.console") as mock_console:
            config.save()

            # verify file was created
            assert config_file.exists()

            # verify content
            with open(config_file) as f:
                saved_config = toml.load(f)

            assert saved_config["test_key"] == "test_value"
            assert saved_config["audio"]["sample_rate"] == 48000

            # verify console output
            mock_console.print.assert_called()
            output = str(mock_console.print.call_args)
            assert "config saved to" in output.lower()

    def test_save_creates_parent_directory(self, temp_dir):
        """test save creates parent directory if needed"""
        config_path = temp_dir / "subdir" / "config.toml"
        config = Config(config_path=config_path)

        config.save()

        assert config_path.exists()
        assert config_path.parent.exists()

    def test_create_default_config_new_file(self, config_file):
        """test creating default config file"""
        config = Config(config_path=config_file)

        with patch("meetcap.utils.config.console") as mock_console:
            config.create_default_config()

            # verify file was created
            assert config_file.exists()

            # verify content is default
            with open(config_file) as f:
                saved_config = toml.load(f)

            assert saved_config["audio"]["sample_rate"] == 48000

            # verify console output
            calls = mock_console.print.call_args_list
            assert len(calls) >= 2
            assert any("created default config" in str(call).lower() for call in calls)
            assert any("edit this file" in str(call).lower() for call in calls)

    def test_create_default_config_existing_file(self, config_file):
        """test create_default_config when file exists"""
        config_file.write_text("existing content")

        config = Config(config_path=config_file)
        original_content = config_file.read_text()

        with patch("meetcap.utils.config.console") as mock_console:
            config.create_default_config()

            # file should not be overwritten
            assert config_file.read_text() == original_content

            # should not print creation message
            calls = mock_console.print.call_args_list
            assert not any("created default config" in str(call).lower() for call in calls)

    def test_expand_path(self):
        """test path expansion"""
        config = Config()

        # test home expansion
        home_path = config.expand_path("~/test/path")
        assert str(home_path).startswith(str(Path.home()))
        assert "test/path" in str(home_path)

        # test environment variable expansion
        with patch.dict(os.environ, {"TEST_VAR": "/custom/path"}):
            env_path = config.expand_path("$TEST_VAR/subdir")
            assert str(env_path) == "/custom/path/subdir"

        # test combined
        with patch.dict(os.environ, {"TEST_VAR": "custom"}):
            combined_path = config.expand_path("~/$TEST_VAR/path")
            assert str(Path.home()) in str(combined_path)
            assert "custom/path" in str(combined_path)

        # test absolute path unchanged
        abs_path = config.expand_path("/absolute/path")
        assert str(abs_path) == "/absolute/path"

    def test_default_config_values(self, tmp_path):
        """test default configuration values"""
        # use a non-existent config file to ensure we're testing defaults
        config_path = tmp_path / "nonexistent" / "config.toml"
        config = Config(config_path)

        # audio defaults
        assert config.get("audio", "preferred_device") == "Aggregate Device"
        assert config.get("audio", "sample_rate") == 48000
        assert config.get("audio", "channels") == 2

        # hotkey defaults
        assert config.get("hotkey", "stop") == "<cmd>+<shift>+s"

        # model defaults
        assert "large-v3" in config.get("models", "stt_model_name")
        assert "Qwen3-4B" in config.get("models", "llm_model_name")

        # paths defaults
        assert "Recordings/meetcap" in config.get("paths", "out_dir")
        assert ".meetcap/models" in config.get("paths", "models_dir")

        # llm defaults
        assert config.get("llm", "n_ctx") == 32768
        assert config.get("llm", "temperature") == 0.4
        assert config.get("llm", "max_tokens") == 4096

        # telemetry defaults
        assert config.get("telemetry", "disable") is True

    def test_config_priority(self, config_file):
        """test configuration priority: env > file > default"""
        # create config file with some overrides
        file_config = {
            "audio": {
                "sample_rate": 44100,
                "channels": 1,
            },
        }
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            toml.dump(file_config, f)

        # set env override for sample_rate only
        with patch.dict(os.environ, {"MEETCAP_SAMPLE_RATE": "96000"}):
            config = Config(config_path=config_file)

        # env should override file
        assert config.get("audio", "sample_rate") == 96000

        # file should override default
        assert config.get("audio", "channels") == 1

        # default should be used for unspecified
        assert config.get("audio", "preferred_device") == "Aggregate Device"


class TestConfigIntegration:
    """integration tests for configuration"""

    def test_full_config_lifecycle(self, temp_dir):
        """test complete configuration lifecycle"""
        config_path = temp_dir / "config.toml"

        # create config
        config = Config(config_path=config_path)

        # modify values
        config.config["custom"] = {"key": "value"}
        config.config["audio"]["sample_rate"] = 96000

        # save
        config.save()

        # load in new instance
        config2 = Config(config_path=config_path)

        # verify persistence
        assert config2.get("custom", "key") == "value"
        assert config2.get("audio", "sample_rate") == 96000

        # verify defaults still work
        assert config2.get("llm", "temperature") == 0.4

    def test_config_with_real_paths(self, temp_dir):
        """test configuration with real path operations"""
        config_path = temp_dir / "config.toml"
        config = Config(config_path=config_path)

        # test path expansion with real directory
        output_dir = temp_dir / "output"
        config.config["paths"]["out_dir"] = str(output_dir)

        expanded = config.expand_path(config.get("paths", "out_dir"))
        assert expanded == output_dir

        # create the directory
        expanded.mkdir(parents=True, exist_ok=True)
        assert expanded.exists()
        assert expanded.is_dir()
