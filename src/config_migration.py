# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
ALICE Configuration Migration

Ensures user config files stay up-to-date across ALICE versions.

When new configuration options are added, this module detects missing keys
in the user's config.yaml and appends them with their default values and
descriptive comments - without overwriting any existing user settings.

The migration runs:
  - On startup (if the config file exists)
  - After a self-update (called by the updater)

Design principles:
  - Never modify existing user values
  - Only add missing keys with defaults
  - Preserve YAML comments and formatting where possible
  - Create a backup before any modification
  - Log all changes clearly
"""

import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ruamel.yaml import YAML as _RuamelYAML
    _HAS_RUAMEL = True
except ImportError:
    _HAS_RUAMEL = False

import yaml

from . import __version__

logger = logging.getLogger(__name__)


def _get_rt_yaml():
    """Return a configured ruamel YAML instance for round-trip (comment-preserving) I/O."""
    yaml_rt = _RuamelYAML()
    yaml_rt.preserve_quotes = True
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    return yaml_rt


def read_config_file(path: Path) -> Any:
    """Read a YAML config file, preserving comments when ruamel.yaml is available.

    Returns a CommentedMap (ruamel) or plain dict (pyyaml fallback).
    """
    if _HAS_RUAMEL:
        try:
            with open(path) as f:
                return _get_rt_yaml().load(f) or {}
        except Exception as e:
            logger.warning("ruamel read failed for %s: %s. Falling back to pyyaml.", path, e)
    with open(path) as f:
        return yaml.safe_load(f) or {}


def write_config_file(path: Path, config: Any) -> None:
    """Write a YAML config file, preserving comments when ruamel.yaml is available.

    Falls back to pyyaml (which strips comments) if ruamel is not installed.
    """
    if _HAS_RUAMEL:
        try:
            with open(path, "w") as f:
                _get_rt_yaml().dump(config, f)
            return
        except Exception as e:
            logger.warning("ruamel write failed for %s: %s. Falling back to pyyaml.", path, e)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


def get_default_config() -> Dict[str, Any]:
    """Generate the full default configuration as a dictionary.

    This mirrors the Pydantic model defaults in config.py but as raw
    dictionaries suitable for YAML serialization. Using the Pydantic models
    directly would lose comments and formatting.
    """
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8080,
            "api_key": None,
            "require_auth": False,
            "registration_mode": "disabled",
            "session_timeout_seconds": 900,
            "block_nsfw": True,
        },
        "models": {
            "directory": "./models",
            "auto_unload_timeout": 0,
            "default_model": None,
            "civitai_api_key": None,
            "huggingface_token": None,
        },
        "generation": {
            "default_steps": 25,
            "default_guidance_scale": 7.5,
            "default_scheduler": "dpm++_sde_karras",
            "max_concurrent": 1,
            "request_timeout": 300,
            "default_width": 512,
            "default_height": 512,
            "backend": "auto",
            "sdcpp_binary": None,
            "sdcpp_threads": 8,
            "force_cpu": False,
            "device_map": None,
            "force_float32": False,
            "force_bfloat16": False,
            "enable_vae_slicing": True,
            "enable_vae_tiling": False,
            "enable_model_cpu_offload": False,
            "enable_sequential_cpu_offload": False,
            "enable_mmap": False,
            "keep_clip_on_cpu": False,
            "attention_slice_size": "auto",
            "vae_decode_cpu": False,
            "enable_torch_compile": False,
            "torch_compile_mode": "reduce-overhead",
            "diffusion_conv_direct": False,
            "vae_conv_direct": True,
            "circular": False,
            "enable_flash_attention": True,
            "cancel_on_disconnect": os.environ.get("ALICE_CANCEL_ON_DISCONNECT", "false").lower()
        in ("true", "1", "yes"),
            "max_cached_models": 2,
            "vram_evict_threshold_gb": 2.0,
            "max_cpu_cached_models": 8,
        },
        "storage": {
            "images_directory": "./images",
            "gallery_file": "./data/gallery.json",
            "auth_directory": "./data/auth",
            "max_storage_gb": 100,
            "retention_days": 7,
            "public_image_expiration_hours": 168,
            "gallery_page_size": 100,
            "audio_directory": "./audio",
        },
        "logging": {
            "level": "WARNING",
            "file": None,
            "max_size_mb": 100,
            "backup_count": 5,
        },
        "model_cache": {
            "enabled": True,
            "database_path": "./data/model_cache.db",
            "sync_on_startup": False,
            "sync_interval_hours": 24,
            "civitai_page_limit": None,
            "huggingface_limit": 10000,
        },
        "audio": {
            "enabled": True,
            "default_model": "stable-audio-open-1.0",
            "default_seconds": 30,
            "default_steps": 100,
            "default_cfg_scale": 7.0,
            "max_concurrent": 1,
            "unload_after_generate": True,
            "request_timeout_seconds": 1800,
            "force_fp32": False,
            "vae_decode_cpu": False,
        },
    }


def find_missing_keys(
    user_config: Dict[str, Any],
    default_config: Dict[str, Any],
    prefix: str = "",
) -> List[Tuple[str, str, Any]]:
    """Find keys present in defaults but missing from user config.

    Args:
        user_config: The user's current configuration.
        default_config: The full default configuration.
        prefix: Dot-separated path prefix for nested keys.

    Returns:
        List of (section, key, default_value) tuples for missing keys.
    """
    missing = []

    for key, default_value in default_config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if key not in user_config:
            missing.append((prefix or "root", key, default_value))
        elif isinstance(default_value, dict) and isinstance(user_config.get(key), dict):
            # Recurse into nested sections
            missing.extend(
                find_missing_keys(user_config[key], default_value, full_key)
            )

    return missing


def migrate_config(
    config_path: Optional[str] = None,
    backup: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Migrate a config file by adding missing keys with defaults.

    Reads the user's config.yaml, compares it against the current defaults,
    and adds any missing keys with their default values. Existing user
    values are never modified.

    Args:
        config_path: Path to the config file. Uses ALICE_CONFIG env var
                     or ./config.yaml if not specified.
        backup: Whether to create a backup before modifying.
        dry_run: If True, only report changes without writing.

    Returns:
        Dictionary with migration results:
        - "migrated": bool - Whether any changes were made
        - "added": list - Keys that were added
        - "config_path": str - Path to the config file
        - "backup_path": str or None - Path to backup if created
    """
    result = {
        "migrated": False,
        "added": [],
        "config_path": None,
        "backup_path": None,
        "version": __version__,
    }

    # Resolve config path
    if config_path is None:
        config_path = os.environ.get("ALICE_CONFIG", "./config.yaml")

    path = Path(config_path)
    result["config_path"] = str(path)

    if not path.exists():
        logger.debug("Config file not found at %s, skipping migration", path)
        return result

    # Load the current config — use ruamel for comment-preserving round-trip
    user_config = None
    yaml_rt = None
    if _HAS_RUAMEL:
        yaml_rt = _RuamelYAML()
        yaml_rt.preserve_quotes = True
        yaml_rt.indent(mapping=2, sequence=4, offset=2)
        try:
            with open(path) as f:
                user_config = yaml_rt.load(f)
                if user_config is None:
                    user_config = {}
        except Exception as e:
            logger.error("Failed to read config file %s: %s", path, e)
            return result
    else:
        logger.warning("ruamel.yaml not installed; config migration will strip comments. Install ruamel.yaml to fix.")
        try:
            with open(path) as f:
                user_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error("Failed to read config file %s: %s", path, e)
            return result

    # Get defaults and find missing keys
    defaults = get_default_config()
    missing = find_missing_keys(user_config, defaults)

    if not missing:
        logger.debug("Config is up to date, no migration needed")
        return result

    # Report what will be added
    for section, key, value in missing:
        result["added"].append(f"{section}.{key}")
        logger.info("Config migration: adding %s.%s = %s", section, key, repr(value))

    if dry_run:
        result["migrated"] = True
        return result

    # Create backup
    if backup:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup_path = path.with_suffix(f".{timestamp}.bak")
        try:
            shutil.copy2(path, backup_path)
            result["backup_path"] = str(backup_path)
            logger.info("Config backup created: %s", backup_path)
        except Exception as e:
            logger.warning("Failed to create config backup: %s", e)

    # Apply missing keys directly to the loaded config object.
    # With ruamel's CommentedMap this preserves existing comments;
    # with a plain dict (pyyaml fallback) it just adds the keys.
    for section, key, value in missing:
        if section == "root":
            user_config[key] = value
        else:
            # Nested key - ensure parent exists
            parts = section.split(".")
            target = user_config
            for part in parts:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[key] = value

    # Write updated config — preserving comments when ruamel is available
    try:
        if yaml_rt is not None:
            # ruamel round-trip: preserves comments, key order, and quoting.
            # We prepend the migration header manually (instead of
            # yaml_set_start_comment which would overwrite existing
            # top-level comments) so user annotations survive.
            with open(path, "w") as f:
                f.write(
                    f"# Configuration migrated to version {__version__}\n"
                    f"# Added {len(missing)} new setting(s) with defaults\n"
                    f"# Original backed up to: {result.get('backup_path', 'N/A')}\n"
                    f"#\n"
                )
                yaml_rt.dump(user_config, f)
        else:
            # Fallback: pyyaml strips comments but still writes valid YAML
            with open(path, "w") as f:
                f.write(f"# Configuration migrated to version {__version__}\n")
                f.write(f"# Added {len(missing)} new setting(s) with defaults\n")
                f.write(f"# Original backed up to: {result.get('backup_path', 'N/A')}\n")
                f.write("#\n")
                yaml.dump(
                    user_config,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )

        result["migrated"] = True
        logger.info(
            "Config migrated: added %d new key(s) to %s",
            len(missing),
            path,
        )
    except Exception as e:
        logger.error("Failed to write migrated config: %s", e)
        # Try to restore backup
        if result["backup_path"]:
            try:
                shutil.copy2(result["backup_path"], path)
                logger.info("Restored config from backup after write failure")
            except Exception:
                logger.error("CRITICAL: Failed to restore config backup!")

    return result


__all__ = ["get_default_config", "find_missing_keys", "migrate_config"]
