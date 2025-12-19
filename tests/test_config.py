# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 The ALICE Authors

"""
ALICE Configuration Tests

Tests for configuration loading and validation.
Run with: pytest tests/test_config.py -v
"""

import pytest
import tempfile
from pathlib import Path


def test_config_loads_defaults():
    """Test configuration loads with default values."""
    from src.config import Config
    
    # Config should have sensible defaults
    config = Config()
    
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 8080
    assert config.generation.default_steps == 25
    assert config.generation.default_guidance_scale == 7.5


def test_config_from_yaml():
    """Test configuration loads from YAML file."""
    from src.config import load_config
    
    yaml_content = """
server:
  host: 127.0.0.1
  port: 9000
  api_key: test-key

models:
  directory: /custom/models

generation:
  default_steps: 30
  default_guidance_scale: 8.0

storage:
  images_directory: /custom/images

logging:
  level: DEBUG
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        
        config = load_config(Path(f.name))
        
        assert config.server.host == "127.0.0.1"
        assert config.server.port == 9000
        assert config.server.api_key == "test-key"
        assert config.generation.default_steps == 30
        assert config.generation.default_guidance_scale == 8.0
        assert config.logging.level == "DEBUG"


def test_config_path_conversion():
    """Test configuration converts paths correctly."""
    from src.config import Config
    
    config = Config()
    
    # Paths should be Path objects
    assert isinstance(config.models.directory, Path)
    assert isinstance(config.storage.images_directory, Path)


def test_config_server_defaults():
    """Test server configuration defaults."""
    from src.config import ServerConfig
    
    server = ServerConfig()
    
    assert server.host == "0.0.0.0"
    assert server.port == 8080
    assert server.api_key is None


def test_config_generation_defaults():
    """Test generation configuration defaults."""
    from src.config import GenerationConfig
    
    gen = GenerationConfig()
    
    assert gen.default_steps == 25
    assert gen.default_guidance_scale == 7.5
    assert gen.default_scheduler == "dpm++_sde_karras"
    assert gen.max_concurrent == 1
    assert gen.request_timeout == 300


def test_config_storage_defaults():
    """Test storage configuration defaults."""
    from src.config import StorageConfig
    
    storage = StorageConfig()
    
    assert storage.max_storage_gb == 100
    assert storage.retention_days == 7


def test_config_logging_defaults():
    """Test logging configuration defaults."""
    from src.config import LoggingConfig
    
    logging = LoggingConfig()
    
    assert logging.level == "INFO"


def test_config_invalid_yaml():
    """Test configuration handles invalid YAML gracefully."""
    from src.config import load_config
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("invalid: yaml: content: [")
        f.flush()
        
        # Should return default config or raise appropriate error
        try:
            config = load_config(Path(f.name))
            # If it returns, should be default config
            assert config is not None
        except Exception as e:
            # If it raises, should be a clear error
            assert "yaml" in str(e).lower() or "parse" in str(e).lower()


def test_config_missing_file():
    """Test configuration handles missing file gracefully."""
    from src.config import load_config
    
    # Should return default config for missing file
    config = load_config(Path("/nonexistent/config.yaml"))
    
    # Should have defaults
    assert config.server.port == 8080


def test_config_partial_yaml():
    """Test configuration merges partial YAML with defaults."""
    from src.config import load_config
    
    yaml_content = """
server:
  port: 9999
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        f.flush()
        
        config = load_config(Path(f.name))
        
        # Custom value
        assert config.server.port == 9999
        # Default values should still be set
        assert config.server.host == "0.0.0.0"
        assert config.generation.default_steps == 25
