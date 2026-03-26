"""
Unit tests for config_loader module.
Tests JSON parsing, default values, save/load roundtrip, and error handling.

Run with: pytest tests/test_config_loader.py -v
"""

import sys
import os
import tempfile
import json
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config_loader import loadConfig, saveConfig, getDefaultConfig, ConfigManager


class TestLoadConfig:
    """Tests for loadConfig function."""

    def test_load_existing_config(self):
        """Test loading a valid config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "camera_index": 1,
                "camera_width": 1280,
                "detection_threshold": 0.75
            }, f)
            temp_path = f.name

        try:
            config = loadConfig(temp_path)
            assert config.camera_index == 1
            assert config.camera_width == 1280
            assert config.detection_threshold == 0.75
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file_returns_defaults(self):
        """Test that missing file returns default config."""
        config = loadConfig("nonexistent_config.json")
        default = getDefaultConfig()
        
        assert config.camera_index == default.camera_index
        assert config.camera_width == default.camera_width
        assert config.detection_threshold == default.detection_threshold

    def test_load_empty_file_returns_defaults(self):
        """Test that empty file returns default config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            config = loadConfig(temp_path)
            default = getDefaultConfig()
            assert config.camera_index == default.camera_index
        finally:
            os.unlink(temp_path)

    def test_partial_config_uses_defaults_for_missing(self):
        """Test that missing values use defaults."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({
                "camera_index": 5
                # Other values missing
            }, f)
            temp_path = f.name

        try:
            config = loadConfig(temp_path)
            assert config.camera_index == 5  # From file
            default = getDefaultConfig()
            assert config.camera_width == default.camera_width  # Default
        finally:
            os.unlink(temp_path)


class TestSaveConfig:
    """Tests for saveConfig function."""

    def test_save_and_load_roundtrip(self):
        """Test that save followed by load preserves values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.json")
            
            # Create config with non-default values
            original = getDefaultConfig()
            original.camera_index = 2
            original.detection_threshold = 0.95
            original.model_path = "custom/model.onnx"
            original.alert_email = "test@example.com"
            
            # Save
            assert saveConfig(config_path, original) is True
            assert os.path.exists(config_path)
            
            # Load and verify
            loaded = loadConfig(config_path)
            assert loaded.camera_index == 2
            assert loaded.detection_threshold == 0.95
            assert loaded.model_path == "custom/model.onnx"
            assert loaded.alert_email == "test@example.com"

    def test_save_creates_directory(self):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "subdir1", "subdir2", "config.json")
            config = getDefaultConfig()
            
            assert saveConfig(nested_path, config) is True
            assert os.path.exists(nested_path)

    def test_save_output_is_valid_json(self):
        """Test that saved file is valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config = getDefaultConfig()
            
            saveConfig(config_path, config)
            
            # Should parse without error
            with open(config_path) as f:
                data = json.load(f)
            
            assert "camera_index" in data
            assert "detection_threshold" in data


class TestDefaultConfig:
    """Tests for default configuration values."""

    def test_default_values_are_sensible(self):
        """Test that default values make sense."""
        default = getDefaultConfig()
        
        # Camera defaults
        assert default.camera_index >= 0
        assert default.camera_width > 0
        assert default.camera_height > 0
        assert default.camera_fps > 0
        
        # Detection defaults
        assert 0.0 < default.detection_threshold < 1.0
        assert default.fall_frame_threshold > 0
        assert default.buffer_size > 0
        
        # Model path should not be empty
        assert default.model_path
        
        # SMTP port should be valid
        assert 0 < default.smtp_port < 65536

    def test_default_config_is_deterministic(self):
        """Test that default config is always the same."""
        default1 = getDefaultConfig()
        default2 = getDefaultConfig()
        
        assert default1.camera_index == default2.camera_index
        assert default1.detection_threshold == default2.detection_threshold
        assert default1.model_path == default2.model_path


class TestConfigManager:
    """Tests for ConfigManager singleton."""

    def test_singleton_instance(self):
        """Test that ConfigManager is a singleton."""
        instance1 = ConfigManager.instance()
        instance2 = ConfigManager.instance()
        assert instance1 is instance2

    def test_load_and_get(self):
        """Test loading and retrieving config."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"camera_index": 42}, f)
            temp_path = f.name

        try:
            manager = ConfigManager.instance()
            manager.load(temp_path)
            
            config = manager.get()
            assert config.camera_index == 42
        finally:
            os.unlink(temp_path)

    def test_set_and_get(self):
        """Test setting and retrieving config."""
        manager = ConfigManager.instance()
        
        new_config = getDefaultConfig()
        new_config.camera_index = 99
        new_config.detection_threshold = 0.99
        
        manager.set(new_config)
        retrieved = manager.get()
        
        assert retrieved.camera_index == 99
        assert retrieved.detection_threshold == 0.99

    def test_get_set_typed_values(self):
        """Test typed value getters and setters."""
        manager = ConfigManager.instance()
        
        # String
        manager.setString("test_string", "hello")
        assert manager.getString("test_string") == "hello"
        assert manager.getString("nonexistent", "default") == "default"
        
        # Int
        manager.setInt("test_int", 42)
        assert manager.getInt("test_int") == 42
        assert manager.getInt("nonexistent", 0) == 0
        
        # Float
        manager.setFloat("test_float", 3.14)
        assert abs(manager.getFloat("test_float") - 3.14) < 0.01
        
        # Bool
        manager.setBool("test_bool", True)
        assert manager.getBool("test_bool") is True
        manager.setBool("test_bool", False)
        assert manager.getBool("test_bool") is False

    def test_invalid_int_returns_default(self):
        """Test that invalid int values return default."""
        manager = ConfigManager.instance()
        manager.setString("bad_int", "not_a_number")
        
        # Should return default, not crash
        result = manager.getInt("bad_int", 100)
        assert result == 100

    def test_invalid_float_returns_default(self):
        """Test that invalid float values return default."""
        manager = ConfigManager.instance()
        manager.setString("bad_float", "not_a_number")
        
        result = manager.getFloat("bad_float", 1.5)
        assert result == 1.5


class TestErrorHandling:
    """Tests for error handling scenarios."""

    def test_load_from_directory_fails_gracefully(self):
        """Test that loading from directory returns defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to "load" a directory
            config = loadConfig(tmpdir)
            default = getDefaultConfig()
            # Should return defaults
            assert config.camera_index == default.camera_index

    def test_save_to_invalid_path(self):
        """Test saving to invalid path."""
        config = getDefaultConfig()
        # Try to save to a path that can't be created
        result = saveConfig("/invalid/path/that/does/not/exist/config.json", config)
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
