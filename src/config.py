# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Configuration Module

Handles loading and managing service configuration from YAML files.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """Server configuration settings."""
    host: str = Field(default="0.0.0.0", description="Server bind address")
    port: int = Field(default=8080, description="Server port")
    api_key: Optional[str] = Field(default=None, description="Optional API key for authentication")
    require_auth: bool = Field(default=False, description="Require authentication for all endpoints")
    registration_mode: str = Field(default="disabled", description="User registration mode: open, invite, approval, disabled")
    session_timeout_seconds: int = Field(default=900, ge=60, le=86400, description="Session timeout in seconds (15 min default)")
    block_nsfw: bool = Field(default=True, description="Block NSFW content generation (admin configurable)")


class ModelsConfig(BaseModel):
    """Model management configuration."""
    directory: Path = Field(default=Path("./models"), description="Models directory path")
    auto_unload_timeout: int = Field(default=300, description="Seconds before unloading idle model")
    default_model: Optional[str] = Field(default=None, description="Default model name")
    civitai_api_key: Optional[str] = Field(default=None, description="CivitAI API key for downloads")
    huggingface_token: Optional[str] = Field(default=None, description="HuggingFace token for private models")


class GenerationConfig(BaseModel):
    """Image generation defaults."""
    default_steps: int = Field(default=25, ge=1, le=150, description="Default inference steps")
    default_guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Default guidance scale")
    default_scheduler: str = Field(default="dpm++_sde_karras", description="Default scheduler")
    max_concurrent: int = Field(default=1, ge=1, description="Max concurrent generations")
    request_timeout: int = Field(default=300, description="Request timeout in seconds")
    default_width: int = Field(default=512, ge=64, le=2048, description="Default image width")
    default_height: int = Field(default=512, ge=64, le=2048, description="Default image height")
    force_cpu: bool = Field(default=False, description="Force CPU mode even if GPU is available")
    device_map: Optional[str] = Field(default=None, description="Device map for model loading (e.g., 'balanced' for AMD APUs)")
    force_float32: bool = Field(default=False, description="Force float32 (required for some AMD GPUs)")
    force_bfloat16: bool = Field(default=False, description="Force bfloat16 (better for AMD Phoenix APU)")
    # Memory optimization settings
    enable_vae_slicing: bool = Field(default=True, description="Enable VAE slicing for lower memory usage")
    enable_vae_tiling: bool = Field(default=False, description="Enable VAE tiling for very large images")
    enable_model_cpu_offload: bool = Field(default=False, description="Enable model CPU offload (slower but uses less VRAM)")
    enable_sequential_cpu_offload: bool = Field(default=False, description="Enable sequential CPU offload (slowest, minimum VRAM)")
    attention_slice_size: Optional[str] = Field(default="auto", description="Attention slice size: 'auto', 'max', or number")
    # AMD gfx1103 workaround: decode VAE on CPU to prevent GPU hang during decode
    vae_decode_cpu: bool = Field(default=False, description="Decode VAE on CPU (fixes GPU hang on AMD gfx1103)")


class StorageConfig(BaseModel):
    """Image storage configuration."""
    images_directory: Path = Field(default=Path("./images"), description="Generated images directory")
    gallery_file: Path = Field(default=Path("./data/gallery.json"), description="Gallery metadata storage file")
    auth_directory: Path = Field(default=Path("./data/auth"), description="Authentication data directory")
    max_storage_gb: int = Field(default=100, ge=1, description="Maximum storage in GB")
    retention_days: int = Field(default=7, ge=1, description="Image retention period in days")
    public_image_expiration_hours: int = Field(default=168, ge=1, description="Default expiration for public images (hours)")
    gallery_page_size: int = Field(default=100, ge=0, description="Number of images to display per page in gallery (use 0 for all)")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="WARNING", description="Log level (WARNING, INFO, DEBUG)")
    file: Optional[Path] = Field(default=None, description="Log file path (null = console only)")
    max_size_mb: int = Field(default=100, description="Max log file size in MB")
    backup_count: int = Field(default=5, description="Number of backup log files")


class ModelCacheConfig(BaseModel):
    """Model catalog caching configuration."""
    enabled: bool = Field(default=True, description="Enable model catalog caching")
    database_path: Path = Field(default=Path("./data/model_cache.db"), description="SQLite database path")
    sync_on_startup: bool = Field(default=False, description="Sync catalogs on server startup (disabled by default to avoid blocking)")
    sync_interval_hours: int = Field(default=24, ge=1, description="Auto-sync interval in hours")
    civitai_page_limit: Optional[int] = Field(default=None, description="Max pages to fetch from CivitAI (null = all)")
    huggingface_limit: int = Field(default=10000, ge=100, description="Max models to fetch from HuggingFace")


class Config(BaseModel):
    """Main configuration container."""
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model_cache: ModelCacheConfig = Field(default_factory=ModelCacheConfig)


def load_config(path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to config file. If None, uses ALICE_CONFIG env var
              or defaults to ./config.yaml
    
    Returns:
        Config object with loaded settings
    """
    if path is None:
        path = os.environ.get("ALICE_CONFIG", "./config.yaml")
    
    config_path = Path(path)
    
    if config_path.exists():
        logger.info("Loading configuration from: %s", config_path)
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
            
            # Convert nested dicts to config objects
            return Config(
                server=ServerConfig(**data.get("server", {})),
                models=ModelsConfig(**data.get("models", {})),
                generation=GenerationConfig(**data.get("generation", {})),
                storage=StorageConfig(**data.get("storage", {})),
                logging=LoggingConfig(**data.get("logging", {})),
                model_cache=ModelCacheConfig(**data.get("model_cache", {}))
            )
        except Exception as e:
            logger.warning("Failed to load config file: %s. Using defaults.", e)
            return Config()
    else:
        logger.info("Config file not found at %s. Using defaults.", config_path)
        return Config()


def setup_logging(config: LoggingConfig) -> None:
    """
    Configure logging based on configuration.
    
    Args:
        config: Logging configuration settings
    """
    # Get log level
    level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Setup handlers - always include console
    handlers = [logging.StreamHandler()]
    
    # Add file handler only if file path is configured
    if config.file:
        try:
            # Create log directory if needed
            log_dir = config.file.parent
            log_dir.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(config.file))
        except Exception as e:
            # If file logging fails, continue with console-only logging
            print(f"Warning: Could not setup file logging: {e}", file=sys.stderr)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    
    # Set level for uvicorn loggers
    logging.getLogger("uvicorn").setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(level)
    
    if config.file:
        logger.info("Logging configured: level=%s, file=%s", config.level, config.file)
    else:
        logger.info("Logging configured: level=%s (console only)", config.level)


# Global config instance - loaded on import
config = load_config()
