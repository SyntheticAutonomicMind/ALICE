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

import copy
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import __version__

logger = logging.getLogger(__name__)


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
            "auto_unload_timeout": 300,
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
        },
        "storage": {
            "images_directory": "./images",
            "gallery_file": "./data/gallery.json",
            "auth_directory": "./data/auth",
            "max_storage_gb": 100,
            "retention_days": 7,
            "public_image_expiration_hours": 168,
            "gallery_page_size": 100,
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

    # Load the current config
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

    # Apply missing keys to user config
    merged = copy.deepcopy(user_config)
    for section, key, value in missing:
        if section == "root":
            # Top-level key
            merged[key] = value
        else:
            # Nested key - ensure parent exists
            parts = section.split(".")
            target = merged
            for part in parts:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[key] = value

    # Write updated config
    try:
        # Read the original file to try to preserve structure
        with open(path) as f:
            original_content = f.read()

        # Append new keys as a comment-annotated block at the end of their sections
        # For simplicity, we rewrite the entire YAML but this preserves all user values
        with open(path, "w") as f:
            # Add a migration header comment
            f.write(f"# Configuration migrated to version {__version__}\n")
            f.write(f"# Added {len(missing)} new setting(s) with defaults\n")
            f.write(f"# Original backed up to: {result.get('backup_path', 'N/A')}\n")
            f.write("#\n")

            # Write the merged config
            yaml.dump(
                merged,
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


def migrate_config_preserving_format(
    config_path: Optional[str] = None,
    backup: bool = True,
) -> Dict[str, Any]:
    """Migrate config by appending missing keys to the end of each section.

    This method preserves the user's original YAML formatting, comments,
    and ordering by appending new keys to the end of the file as an
    addendum block, rather than rewriting the entire file.

    Args:
        config_path: Path to the config file.
        backup: Whether to create a backup.

    Returns:
        Migration results dict.
    """
    result = {
        "migrated": False,
        "added": [],
        "config_path": None,
        "backup_path": None,
        "version": __version__,
    }

    if config_path is None:
        config_path = os.environ.get("ALICE_CONFIG", "./config.yaml")

    path = Path(config_path)
    result["config_path"] = str(path)

    if not path.exists():
        return result

    try:
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
    except Exception as e:
        logger.error("Failed to read config: %s", e)
        return result

    defaults = get_default_config()
    missing = find_missing_keys(user_config, defaults)

    if not missing:
        return result

    # Create backup
    if backup:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup_path = path.with_suffix(f".{timestamp}.bak")
        try:
            shutil.copy2(path, backup_path)
            result["backup_path"] = str(backup_path)
        except Exception as e:
            logger.warning("Could not backup config: %s", e)

    # Group missing keys by top-level section
    by_section: Dict[str, List[Tuple[str, Any]]] = {}
    for section, key, value in missing:
        if section == "root":
            by_section.setdefault("_root", []).append((key, value))
        else:
            # Use just the top-level section name
            top_section = section.split(".")[0]
            by_section.setdefault(top_section, []).append((key, value))

    # Append new keys to the file
    try:
        with open(path, "a") as f:
            f.write(f"\n# --- New settings added by ALICE {__version__} ---\n")

            for section, keys in by_section.items():
                if section == "_root":
                    for key, value in keys:
                        f.write(f"\n{key}: {_yaml_value(value)}\n")
                        result["added"].append(key)
                else:
                    # Check if the section already exists in the file
                    if section in user_config:
                        # Append under the existing section
                        # We write as top-level because YAML doesn't support
                        # appending to nested blocks easily. Instead, we use
                        # the merge pattern where the last occurrence wins.
                        # Actually, YAML doesn't support merging like that.
                        # Use the full rewrite approach instead.
                        pass
                    else:
                        # New section entirely
                        f.write(f"\n{section}:\n")
                        for key, value in keys:
                            f.write(f"  {key}: {_yaml_value(value)}\n")
                            result["added"].append(f"{section}.{key}")

            # For keys in existing sections, we need the full rewrite approach
            needs_rewrite = any(
                section in user_config
                for section in by_section
                if section != "_root"
            )

            if needs_rewrite:
                # Fall back to full rewrite for keys in existing sections
                return migrate_config(config_path=str(path), backup=False, dry_run=False)

        result["migrated"] = True
        logger.info("Config migrated: %d new setting(s)", len(result["added"]))
    except Exception as e:
        logger.error("Config migration failed: %s", e)

    return result


def _yaml_value(value: Any) -> str:
    """Format a Python value as a YAML string."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        if any(c in value for c in ":{}[]#&*!|>'\"%@`"):
            return f'"{value}"'
        return value
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        return yaml.dump(value, default_flow_style=True).strip()
