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

from src.config import (
    ServerConfig, ModelsConfig, GenerationConfig,
    StorageConfig, LoggingConfig, ModelCacheConfig, AudioConfig,
)


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
    
    assert logging.level == "WARNING"


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


def test_config_cancel_on_disconnect_default(monkeypatch):
    """Test cancel_on_disconnect defaults to False."""
    monkeypatch.delenv("ALICE_CANCEL_ON_DISCONNECT", raising=False)
    from src.config import GenerationConfig
    gen = GenerationConfig()
    assert gen.cancel_on_disconnect is False


@pytest.mark.parametrize("env_val,expected", [
    ("true", True),
    ("1", True),
    ("yes", True),
    ("TRUE", True),
    ("True", True),
    ("false", False),
    ("0", False),
    ("no", False),
    ("FALSE", False),
])
def test_config_cancel_on_disconnect_env_var(monkeypatch, env_val, expected):
    """Test cancel_on_disconnect overridden by ALICE_CANCEL_ON_DISCONNECT env var."""
    from src.config import GenerationConfig
    monkeypatch.setenv("ALICE_CANCEL_ON_DISCONNECT", env_val)
    gen = GenerationConfig()
    assert gen.cancel_on_disconnect is expected


def test_config_cancel_on_disconnect_yaml(monkeypatch, tmp_path):
    """Test cancel_on_disconnect loads from YAML configuration."""
    monkeypatch.delenv("ALICE_CANCEL_ON_DISCONNECT", raising=False)
    from src.config import load_config

    config_file = tmp_path / "config.yaml"
    config_file.write_text("generation:\n  cancel_on_disconnect: true\n", encoding="utf-8")

    config = load_config(config_file)
    assert config.generation.cancel_on_disconnect is True


def test_config_migration_sync():
    """Test that get_default_config in config_migration stays in sync with GenerationConfig."""
    from src.config import GenerationConfig, AudioConfig, ServerConfig, ModelsConfig, StorageConfig, LoggingConfig, ModelCacheConfig
    from src.config_migration import get_default_config

    defaults = get_default_config()
    gen_defaults = defaults.get("generation", {})

    # Verify cancel_on_disconnect specifically
    assert "cancel_on_disconnect" in gen_defaults
    assert gen_defaults["cancel_on_disconnect"] is False

    # Invariant: every field in GenerationConfig must exist in migration defaults
    model_fields = getattr(GenerationConfig, "model_fields", None) or getattr(GenerationConfig, "__fields__", {})
    missing_fields = set(model_fields.keys()) - set(gen_defaults.keys())
    assert not missing_fields, f"Missing fields in config_migration.py get_default_config(): {missing_fields}"


@pytest.mark.parametrize("model_cls,section", [
    (ServerConfig, "server"),
    (ModelsConfig, "models"),
    (GenerationConfig, "generation"),
    (StorageConfig, "storage"),
    (LoggingConfig, "logging"),
    (ModelCacheConfig, "model_cache"),
    (AudioConfig, "audio"),
])
def test_config_defaults_sync(model_cls, section):
    """Test that every Pydantic field default matches get_default_config() value.

    This catches drift between config.py (Pydantic models) and
    config_migration.py (get_default_config). Fields with default_factory
    or env-var-based defaults are skipped.
    """
    from src.config_migration import get_default_config
    defaults = get_default_config()
    section_defaults = defaults.get(section, {})
    fields = getattr(model_cls, "model_fields", None) or getattr(model_cls, "__fields__", {})

    for fname, field_info in fields.items():
        if fname not in section_defaults:
            continue  # Key-existence is checked by test_config_migration_sync
        # Skip fields with default_factory (e.g. cancel_on_disconnect reads env var)
        if field_info.default_factory is not None:
            continue
        pydantic_default = field_info.default
        migration_default = section_defaults[fname]
        # Path defaults in config.py vs strings in config_migration — normalize
        from pathlib import Path
        if isinstance(pydantic_default, Path) and not isinstance(migration_default, Path):
            migration_default = Path(migration_default)
        # None == None, True == True, etc.
        assert pydantic_default == migration_default, (
            f"Default mismatch in {section}.{fname}: "
            f"config.py={pydantic_default!r} vs config_migration.py={migration_default!r}"
        )


def test_audio_timeout_default():
    """Test that the audio request_timeout_seconds default is sufficient for music generation."""
    from src.config import AudioConfig
    audio = AudioConfig()
    assert audio.request_timeout_seconds == 1800  # 30 min — covers MiniMax-Music3 full songs


def test_config_migration_preserves_comments(tmp_path):
    """Test that migrate_config preserves YAML comments."""
    from src.config_migration import migrate_config
    from ruamel.yaml import YAML

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "# My custom config\n"
        "# This is a comment I want to keep\n"
        "server:\n"
        "  host: 0.0.0.0\n"
        "  port: 8090\n"
        "generation:\n"
        "  default_steps: 25\n"
        "  # My custom generation settings\n"
        "  force_float32: true\n",
        encoding="utf-8",
    )

    result = migrate_config(config_path=str(config_file), backup=True)
    assert result["migrated"]

    # Read the file back and verify comments are preserved
    content = config_file.read_text(encoding="utf-8")
    assert "My custom config" in content, "Top-level comment was stripped!"
    assert "This is a comment I want to keep" in content, "Second comment was stripped!"
    assert "My custom generation settings" in content, "Inline section comment was stripped!"
    # Verify user values are preserved
    assert "port: 8090" in content
    assert "force_float32: true" in content


