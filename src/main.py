# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Main Application

FastAPI application providing OpenAI-compatible endpoints for
Stable Diffusion image generation.
"""

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Header, Depends, Request, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse

# Configure PyTorch threading BEFORE importing torch/diffusers
# This must happen before any torch imports to take effect
import multiprocessing
num_cpus = multiprocessing.cpu_count()
os.environ.setdefault("OMP_NUM_THREADS", str(num_cpus))
os.environ.setdefault("MKL_NUM_THREADS", str(num_cpus))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(num_cpus))

from .config import config, setup_logging
from .model_registry import ModelRegistry
from .generator import GeneratorService
from .downloader import DownloadManager
from .model_cache import ModelCacheService
from .auth import AuthManager, SESSION_TIMEOUT_SECONDS, SESSION_INACTIVITY_TIMEOUT_SECONDS, set_session_timeout
from .gallery import GalleryManager, ImageRecord
from .cancellation import get_cancellation_registry, CancellationError
from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ResponseMessage,
    ImageMetadata,
    Usage,
    ModelsResponse,
    ModelInfo,
    LoRAsResponse,
    LoRAInfo,
    HealthResponse,
    MetricsResponse,
    ErrorResponse,
    ErrorDetail,
    CivitAISearchRequest,
    CivitAIDownloadRequest,
    HuggingFaceSearchRequest,
    HuggingFaceDownloadRequest,
    DirectDownloadRequest,
    DownloadTaskInfo,
    DownloadListResponse,
    CreateAPIKeyRequest,
    CreateInviteRequest,
    APIKeyInfo,
    APIKeyCreatedResponse,
    APIKeyListResponse,
    SessionInfo,
    SessionCreatedResponse,
    SessionListResponse,
    AuthStatsResponse,
    GalleryImageInfo,
    GalleryListResponse,
    UpdateImagePrivacyRequest,
    GalleryStatsResponse,
)

# Setup logging
setup_logging(config.logging)
logger = logging.getLogger(__name__)

# NOTE: PyTorch threading configuration removed - moved to PyTorchBackend
# Importing torch at module level causes hangs on some AMD GPUs (gfx90c)
# The PyTorch backend handles this configuration internally now.

# =============================================================================
# SERVICE INITIALIZATION
# =============================================================================

# Initialize services (will be configured on startup)
model_registry: Optional[ModelRegistry] = None
generator: Optional[GeneratorService] = None
download_manager: Optional[DownloadManager] = None
model_cache_service: Optional[ModelCacheService] = None
auth_manager: Optional[AuthManager] = None
gallery_manager: Optional[GalleryManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global model_registry, generator, download_manager, model_cache_service, auth_manager, gallery_manager
    
    logger.info("ALICE starting up...")
    
    # Configure session timeout from config
    set_session_timeout(config.server.session_timeout_seconds)
    
    # Initialize services
    model_registry = ModelRegistry(config.models.directory)
    generator = GeneratorService(
        images_dir=config.storage.images_directory,
        backend_name=config.generation.backend,
        sdcpp_binary=config.generation.sdcpp_binary,
        sdcpp_threads=config.generation.sdcpp_threads,
        default_steps=config.generation.default_steps,
        default_guidance_scale=config.generation.default_guidance_scale,
        default_scheduler=config.generation.default_scheduler,
        default_width=config.generation.default_width,
        default_height=config.generation.default_height,
        force_cpu=config.generation.force_cpu,
        device_map=config.generation.device_map,
        force_float32=config.generation.force_float32,
        force_bfloat16=config.generation.force_bfloat16,
        enable_vae_slicing=config.generation.enable_vae_slicing,
        enable_vae_tiling=config.generation.enable_vae_tiling,
        enable_model_cpu_offload=config.generation.enable_model_cpu_offload,
        enable_sequential_cpu_offload=config.generation.enable_sequential_cpu_offload,
        attention_slice_size=config.generation.attention_slice_size,
        vae_decode_cpu=config.generation.vae_decode_cpu,
        enable_torch_compile=config.generation.enable_torch_compile,
        torch_compile_mode=config.generation.torch_compile_mode,
        max_concurrent_generations=config.generation.max_concurrent,
    )
    download_manager = DownloadManager(
        models_dir=config.models.directory,
        loras_dir=config.models.directory / "loras",
        max_concurrent=2,
        civitai_api_key=config.models.civitai_api_key,
        huggingface_token=config.models.huggingface_token,
    )
    
    # Initialize model cache service if enabled
    logger.info("Checking model cache config: enabled=%s", config.model_cache.enabled)
    if config.model_cache.enabled:
        logger.info("Model cache is enabled, initializing...")
        try:
            model_cache_service = ModelCacheService(
                database_path=config.model_cache.database_path,
                civitai_api_key=config.models.civitai_api_key,
                huggingface_token=config.models.huggingface_token,
                civitai_page_limit=config.model_cache.civitai_page_limit,
                huggingface_limit=config.model_cache.huggingface_limit,
            )
            logger.info("Model cache service initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize model cache service: %s", e, exc_info=True)
            model_cache_service = None
    else:
        logger.warning("Model cache is DISABLED in configuration")
    
    auth_manager = AuthManager(
        data_dir=config.storage.auth_directory
    )
    gallery_manager = GalleryManager(
        storage_path=config.storage.gallery_file
    )
    
    # Start download manager
    await download_manager.start()
    
    # Ensure directories exist
    config.storage.images_directory.mkdir(parents=True, exist_ok=True)
    config.models.directory.mkdir(parents=True, exist_ok=True)
    
    # NOTE: /images/ endpoint is defined below with authentication
    # Images are NOT mounted as static files to enforce access control
    
    # Web interface - serve static assets (CSS, JS) without auth
    # Pages are served via custom endpoints with auth checks
    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.exists():
        # Mount style.css and app.js as static files (no auth needed for these)
        @app.get("/web/style.css")
        async def serve_style():
            return FileResponse(web_dir / "style.css", media_type="text/css")
        
        @app.get("/web/app.js")
        async def serve_app_js():
            return FileResponse(web_dir / "app.js", media_type="application/javascript")
        
        # Serve logo image
        @app.get("/web/alice-logo.png")
        async def serve_logo():
            return FileResponse(web_dir / "alice-logo.png", media_type="image/png")
        
        # Serve fonts directory for offline use
        @app.get("/web/fonts/{filename}")
        async def serve_font(filename: str):
            font_path = web_dir / "fonts" / filename
            if not font_path.exists():
                raise HTTPException(status_code=404, detail="Font file not found")
            # Determine media type based on file extension
            media_type = "font/ttf" if filename.endswith(".ttf") else "text/css"
            return FileResponse(font_path, media_type=media_type)
        
        logger.info("Web interface endpoints configured at /web")
    else:
        logger.warning("Web directory not found at %s", web_dir)
    
    # Scan for models
    models = model_registry.scan_models()
    logger.info("Found %d models", len(models))
    
    # Start model cache sync if enabled
    sync_task = None
    if model_cache_service and config.model_cache.sync_on_startup:
        async def model_cache_sync_task():
            """Background task for periodic model catalog synchronization."""
            # Initial sync
            try:
                logger.info("Starting initial model catalog sync...")
                await model_cache_service.sync_civitai()
                await model_cache_service.sync_huggingface()
                logger.info("Initial catalog sync complete")
            except Exception as e:
                logger.error("Initial catalog sync failed: %s", e)
            
            # Periodic sync
            while True:
                try:
                    await asyncio.sleep(config.model_cache.sync_interval_hours * 3600)
                    logger.info("Starting scheduled model catalog sync...")
                    await model_cache_service.sync_civitai()
                    await model_cache_service.sync_huggingface()
                    logger.info("Scheduled catalog sync complete")
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Scheduled catalog sync failed: %s", e)
        
        sync_task = asyncio.create_task(model_cache_sync_task())
    
    # Start background task for cleaning up expired images
    async def cleanup_expired_images_task():
        """Background task to periodically clean up expired images."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                if gallery_manager:
                    cleaned = gallery_manager.cleanup_expired(config.storage.images_directory)
                    if cleaned > 0:
                        logger.info("Cleaned up %d expired images", cleaned)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in cleanup task: %s", e)
    
    cleanup_task = asyncio.create_task(cleanup_expired_images_task())
    
    logger.info(
        "ALICE ready on http://%s:%d",
        config.server.host,
        config.server.port
    )
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("ALICE shutting down...")
    
    # Cancel background tasks
    cleanup_task.cancel()
    if sync_task:
        sync_task.cancel()
    
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    
    if sync_task:
        try:
            await sync_task
        except asyncio.CancelledError:
            pass
    
    # Stop download manager
    if download_manager:
        await download_manager.stop()
    
    # Unload model
    if generator and generator.is_model_loaded:
        await generator.unload_model()
    
    logger.info("ALICE shutdown complete")


# =============================================================================
# APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="ALICE",
    description="Remote Stable Diffusion Service with OpenAI-compatible API",
    version="1.2.1",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware - MUST be added before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# WEB PAGE ENDPOINTS (Auth-protected)
# =============================================================================

# Helper to get web directory path
def _get_web_dir() -> Path:
    return Path(__file__).parent.parent / "web"


def _verify_web_auth_cookie(alice_session: Optional[str]) -> bool:
    """Verify if request has valid session cookie for protected pages."""
    if auth_manager is None:
        return False
    
    if not alice_session:
        return False
    
    # The session cookie contains the API key
    key_record = auth_manager.verify_api_key(alice_session)
    return key_record is not None


def _verify_web_admin_cookie(alice_session: Optional[str]) -> bool:
    """Verify if request has valid admin session cookie."""
    # If auth is disabled, allow access
    if not config.server.require_auth:
        return True
    
    if auth_manager is None:
        return False
    
    if not alice_session:
        return False
    
    # The session cookie contains the API key
    key_record = auth_manager.verify_api_key(alice_session)
    return key_record is not None and key_record.is_admin


# Public pages - no auth required
@app.get("/web/login.html", response_class=HTMLResponse)
async def serve_login():
    """Serve the login page - always accessible."""
    web_dir = _get_web_dir()
    if not (web_dir / "login.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(web_dir / "login.html", media_type="text/html")


@app.get("/web/debug-cookies.html", response_class=HTMLResponse)
async def serve_debug_cookies():
    """Serve the cookie debug page - always accessible for debugging."""
    web_dir = _get_web_dir()
    if not (web_dir / "debug-cookies.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(web_dir / "debug-cookies.html", media_type="text/html")


@app.get("/web/emergency-admin.html", response_class=HTMLResponse)
async def serve_emergency_admin():
    """Serve the emergency admin access page - always accessible for debugging."""
    web_dir = _get_web_dir()
    if not (web_dir / "emergency-admin.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    return FileResponse(web_dir / "emergency-admin.html", media_type="text/html")


@app.get("/web/generate.html", response_class=HTMLResponse)
async def serve_generate(alice_session: Optional[str] = Cookie(None)):
    """Serve the generate page - requires authentication."""
    web_dir = _get_web_dir()
    if not (web_dir / "generate.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    # If server requires auth, check for valid session cookie
    if config.server.require_auth and not _verify_web_auth_cookie(alice_session):
        return RedirectResponse(url="/web/login.html?return=/web/generate.html", status_code=303)
    
    return FileResponse(web_dir / "generate.html", media_type="text/html")


# Protected pages - require authentication via session cookie
@app.get("/web/", response_class=HTMLResponse)
@app.get("/web/index.html", response_class=HTMLResponse)
async def serve_index(alice_session: Optional[str] = Cookie(None)):
    """Serve the main dashboard - ALWAYS requires admin authentication."""
    web_dir = _get_web_dir()
    if not (web_dir / "index.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Dashboard is admin-only - ALWAYS require admin auth
    if not _verify_web_admin_cookie(alice_session):
        return RedirectResponse(url="/web/login.html?return=/web/", status_code=303)
    
    return FileResponse(web_dir / "index.html", media_type="text/html")


@app.get("/web/admin.html", response_class=HTMLResponse)
async def serve_admin(alice_session: Optional[str] = Cookie(None)):
    """Serve the admin page - ALWAYS requires admin authentication."""
    web_dir = _get_web_dir()
    if not (web_dir / "admin.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Admin page is admin-only - ALWAYS require admin auth
    if not _verify_web_admin_cookie(alice_session):
        return RedirectResponse(url="/web/login.html?return=/web/admin.html", status_code=303)
    
    return FileResponse(web_dir / "admin.html", media_type="text/html")


@app.get("/web/models.html", response_class=HTMLResponse)
async def serve_models(alice_session: Optional[str] = Cookie(None)):
    """Serve the models page - ALWAYS requires admin authentication."""
    web_dir = _get_web_dir()
    if not (web_dir / "models.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Models page is admin-only - ALWAYS require admin auth
    if not _verify_web_admin_cookie(alice_session):
        return RedirectResponse(url="/web/login.html?return=/web/models.html", status_code=303)
    
    return FileResponse(web_dir / "models.html", media_type="text/html")


@app.get("/web/download.html", response_class=HTMLResponse)
async def serve_download(alice_session: Optional[str] = Cookie(None)):
    """Serve the download page - ALWAYS requires admin authentication."""
    web_dir = _get_web_dir()
    if not (web_dir / "download.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Download page is admin-only - ALWAYS require admin auth
    if not _verify_web_admin_cookie(alice_session):
        return RedirectResponse(url="/web/login.html?return=/web/download.html", status_code=303)
    
    return FileResponse(web_dir / "download.html", media_type="text/html")


@app.get("/web/gallery.html", response_class=HTMLResponse)
async def serve_gallery(alice_session: Optional[str] = Cookie(None)):
    """Serve the gallery page - requires authentication."""
    web_dir = _get_web_dir()
    if not (web_dir / "gallery.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    # If server requires auth, check for valid session cookie
    if config.server.require_auth and not _verify_web_auth_cookie(alice_session):
        return RedirectResponse(url="/web/login.html?return=/web/gallery.html", status_code=303)
    
    return FileResponse(web_dir / "gallery.html", media_type="text/html")


@app.get("/web/prompting.html", response_class=HTMLResponse)
async def serve_prompting(alice_session: Optional[str] = Cookie(None)):
    """Serve the prompting guide page - requires authentication."""
    web_dir = _get_web_dir()
    if not (web_dir / "prompting.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    # If server requires auth, check for valid session cookie
    if config.server.require_auth and not _verify_web_auth_cookie(alice_session):
        return RedirectResponse(url="/web/login.html?return=/web/prompting.html", status_code=303)
    
    return FileResponse(web_dir / "prompting.html", media_type="text/html")


@app.get("/web/debug.html", response_class=HTMLResponse)
async def serve_debug(alice_session: Optional[str] = Cookie(None)):
    """Serve the debug page - ALWAYS requires admin authentication."""
    web_dir = _get_web_dir()
    if not (web_dir / "debug.html").exists():
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Debug page is admin-only - ALWAYS require admin auth
    if not _verify_web_admin_cookie(alice_session):
        return RedirectResponse(url="/web/login.html?return=/web/debug.html", status_code=303)
    
    return FileResponse(web_dir / "debug.html", media_type="text/html")


# =============================================================================
# AUTHENTICATION
# =============================================================================

from .auth import AccessLevel

async def get_access_level(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
) -> AccessLevel:
    """
    Determine the access level for the current request.
    
    Checks both Authorization header and X-Api-Key header.
    
    Returns:
        AccessLevel: ADMIN, USER, or ANONYMOUS
    """
    # No authentication configured and require_auth is off - everyone gets user access
    if not config.server.api_key and auth_manager is None and not config.server.require_auth:
        return AccessLevel.USER
    
    # Try to extract API key from headers
    api_key = None
    
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    elif x_api_key:
        api_key = x_api_key
    
    if not api_key:
        # No key provided
        # If require_auth is disabled, give user access (not anonymous)
        if not config.server.require_auth:
            return AccessLevel.USER
        return AccessLevel.ANONYMOUS  # Will be blocked by require_access_level
    
    # Check simple config API key (gives user access)
    if config.server.api_key and api_key == config.server.api_key:
        return AccessLevel.USER
    
    # Check registered API keys
    if auth_manager:
        key_record = auth_manager.verify_api_key(api_key)
        if key_record:
            return key_record.get_access_level()
    
    # Invalid key - anonymous access if allowed
    return AccessLevel.ANONYMOUS


async def verify_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """
    Legacy API key verification for backwards compatibility.
    Always returns True - use get_access_level for proper auth.
    """
    # For backwards compatibility, always return True
    # Individual endpoints should use require_access_level instead
    return True


def require_access_level(minimum_level: AccessLevel):
    """
    Dependency that requires a minimum access level.
    
    If config.server.require_auth is True, anonymous access is not allowed
    for any endpoint using this dependency.
    
    Usage:
        @app.get("/admin-only")
        async def admin_endpoint(access: AccessLevel = Depends(require_access_level(AccessLevel.ADMIN))):
            ...
    """
    async def check_access(
        authorization: Optional[str] = Header(None),
        x_api_key: Optional[str] = Header(None),
    ) -> AccessLevel:
        level = await get_access_level(authorization, x_api_key)
        
        # If require_auth is enabled, anonymous access is not allowed
        if config.server.require_auth and level == AccessLevel.ANONYMOUS:
            raise HTTPException(
                status_code=401,
                detail="Authentication required. Use 'Authorization: Bearer <key>' or 'X-Api-Key' header."
            )
        
        # Check if access level is sufficient
        level_order = {
            AccessLevel.ANONYMOUS: 0,
            AccessLevel.USER: 1,
            AccessLevel.ADMIN: 2,
        }
        
        if level_order[level] < level_order[minimum_level]:
            if level == AccessLevel.ANONYMOUS:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required. Use 'Authorization: Bearer <key>' header."
                )
            else:
                raise HTTPException(
                    status_code=403,
                    detail=f"Insufficient permissions. Requires {minimum_level.value} access."
                )
        
        return level
    
    return check_access


class AnonymousUser:
    """Represents an anonymous user when require_auth is disabled."""
    id: str = "anonymous"
    name: str = "Anonymous"
    is_admin: bool = False
    
    def get_access_level(self):
        return AccessLevel.USER


async def get_current_user(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    alice_session: Optional[str] = Cookie(None),
):
    """
    Dependency that returns the current authenticated user (APIKey record).
    
    If require_auth is False and no credentials provided, returns AnonymousUser.
    Otherwise raises HTTPException if not authenticated.
    """
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Try session cookie first (contains raw API key)
    if alice_session:
        api_key = auth_manager.verify_api_key(alice_session)
        if api_key:
            return api_key
    
    # Try API key from headers
    key_string = None
    if authorization and authorization.startswith("Bearer "):
        key_string = authorization[7:]
    elif x_api_key:
        key_string = x_api_key
    
    if key_string:
        api_key = auth_manager.verify_api_key(key_string)
        if api_key:
            return api_key
    
    # No valid credentials - check if auth is required
    if not config.server.require_auth:
        return AnonymousUser()


def check_nsfw_content(text: str) -> bool:
    """
    Check if text contains NSFW keywords or content.
    
    Comprehensive filter that detects:
    - Direct NSFW keywords
    - Common variations and euphemisms
    - Obfuscation attempts (leetspeak, spacing, symbols)
    - Medical/technical terms used inappropriately
    - Context-based detection
    
    Returns True if NSFW content detected, False otherwise.
    
    Note: Filter lists are stored in Base64-encoded format to keep the source code clean
    while maintaining full filtering functionality.
    """
    import re
    import base64
    import json
    
    # Base64-encoded filter lists (obfuscated for code cleanliness)
    # These are decoded at runtime for filtering
    NSFW_KEYWORDS_B64 = 'WyJudWRlIiwgIm5ha2VkIiwgIm5zZnciLCAicG9ybiIsICJwb3Jub2dyYXBoaWMiLCAiZXJvdGljIiwgInh4eCIsICJzZXgiLCAic2V4dWFsIiwgInNleHVhbGl0eSIsICJleHBsaWNpdCIsICJhZHVsdCBvbmx5IiwgIjE4KyIsICJyMTgiLCAidG9wbGVzcyIsICJib3R0b21sZXNzIiwgIm5pcHBsZSIsICJuaXBwbGVzIiwgImJyZWFzdHMiLCAiZ2VuaXRhbGlhIiwgImdlbml0YWwiLCAicGVuaXMiLCAidmFnaW5hIiwgInZ1bHZhIiwgImFudXMiLCAiY29jayIsICJkaWNrIiwgInB1c3N5IiwgImN1bnQiLCAiYXNzIiwgInRpdHMiLCAiYm9vYnMiLCAibGluZ2VyaWUiLCAidW5kZXJ3ZWFyIiwgInBhbnRpZXMiLCAiYnJhIiwgInRob25nIiwgImctc3RyaW5nIiwgImJpa2luaSBib3R0b20iLCAic2VlLXRocm91Z2giLCAidHJhbnNwYXJlbnQgY2xvdGhpbmciLCAiaW50ZXJjb3Vyc2UiLCAiY29pdHVzIiwgImNvcHVsYXRpb24iLCAibWF0aW5nIiwgImZlbGxhdGlvIiwgImN1bm5pbGluZ3VzIiwgIm9yYWwgc2V4IiwgImJsb3dqb2IiLCAiYmxvdyBqb2IiLCAiaGFuZGpvYiIsICJoYW5kIGpvYiIsICJmaW5nZXJpbmciLCAicGVuZXRyYXRpb24iLCAicGVuZXRyYXRpbmciLCAibWFzdHVyYmF0aW9uIiwgIm1hc3R1cmJhdGluZyIsICJvcmdhc20iLCAiY2xpbWF4IiwgImVqYWN1bGF0aW9uIiwgImFyb3VzYWwiLCAiYXJvdXNlZCIsICJlcmVjdGlvbiIsICJoYXJkLW9uIiwgIndldCIsICJ0aHJlZXNvbWUiLCAiZ2FuZ2JhbmciLCAiZ2FuZyBiYW5nIiwgIm9yZ3kiLCAiYnVra2FrZSIsICJiZHNtIiwgImJvbmRhZ2UiLCAiZG9taW5hdGlvbiIsICJzdWJtaXNzaW9uIiwgInNhZGlzbSIsICJtYXNvY2hpc20iLCAiZmV0aXNoIiwgImtpbmsiLCAia2lua3kiLCAic2VkdWN0aXZlIiwgInByb3ZvY2F0aXZlIiwgInN1Z2dlc3RpdmUiLCAic3VsdHJ5IiwgInNlbnN1YWwiLCAibGV3ZCIsICJsYXNjaXZpb3VzIiwgImx1c3RmdWwiLCAiaG9ybnkiLCAicmFuZHkiLCAiaW50aW1hdGUiLCAiaW50aW1hY3kiLCAicGFzc2lvbmF0ZSIsICJoZW50YWkiLCAiZG91amluIiwgImVjY2hpIiwgImFoZWdhbyIsICJ5YW9pIiwgInl1cmkiLCAicnVsZTM0IiwgInJ1bGUgMzQiLCAibnNmbCIsICJhZHVsdCBhY3Rpdml0eSIsICJiZWRyb29tIHNjZW5lIiwgImhvcml6b250YWwiLCAibWFraW5nIGxvdmUiLCAic2xlZXBpbmcgdG9nZXRoZXIiLCAibmV0ZmxpeCBhbmQgY2hpbGwiLCAic3RlYW15IiwgInNwaWN5IiwgInNhdWN5IiwgIm5hdWdodHkiLCAiZGlydHkiLCAiZnVjayIsICJmdWNraW5nIiwgImZ1Y2tlZCIsICJzY3Jld2luZyIsICJiYW5naW5nIiwgImh1bXBpbmciLCAiZ3JpbmRpbmciLCAicmlkaW5nIiwgImRpbGRvIiwgInZpYnJhdG9yIiwgInNleCB0b3kiLCAiYnV0dHBsdWciLCAiYnV0dCBwbHVnIiwgImFuYWwiLCAidmFnaW5hbCIsICJjdW0iLCAiY3VtbWluZyIsICJzZW1lbiIsICJjcmVhbXBpZSIsICJmYWNpYWwiLCAic3F1aXJ0IiwgInNxdWlydGluZyIsICJsYWN0YXRpb24iLCAibGFjdGF0aW5nIiwgImluY2VzdCIsICJsb2xpIiwgImxvbGljb24iLCAic2hvdGEiLCAic2hvdGFjb24iLCAicGVkb3BoaWxlIiwgInJhcGUiLCAicmFwaW5nIiwgIm1vbGVzdCIsICJhc3NhdWx0IiwgImxhYmlhIiwgImNsaXRvcmlzIiwgInRlc3RpY2xlcyIsICJzY3JvdHVtIiwgInByb3N0YXRlIiwgImVyb2dlbm91cyIsICJtYW1tYXJ5IiwgInBoYWxsdXMiLCAiZm9yZXNraW4iXQ=='
    
    # Decode the filter lists
    nsfw_keywords = json.loads(base64.b64decode(NSFW_KEYWORDS_B64).decode())
    
    # Normalize text for checking
    text_lower = text.lower().strip()
    
    # Remove extra spaces for detection
    text_normalized = re.sub(r'\s+', ' ', text_lower)
    
    # Check direct keywords with word boundaries to avoid false positives
    for keyword in nsfw_keywords:
        # Use word boundary matching for better accuracy
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text_normalized):
            logger.warning("NSFW content blocked: keyword detected in prompt")
            return True
    
    # Obfuscation patterns (Base64-encoded)
    OBFUSCATION_PATTERNS_B64 = 'W1sibltcXFdfXSp1W1xcV19dKmRbXFxXX10qZSIsICJudWRlIl0sIFsic1tcXFdfXSplW1xcV19dKngiLCAic2V4Il0sIFsicFtcXFdfXSpvW1xcV19dKnJbXFxXX10qbiIsICJwb3JuIl0sIFsibltcXFdfXSphW1xcV19dKmtbXFxXX10qZVtcXFdfXSpkIiwgIm5ha2VkIl0sIFsiZltcXFdfXSp1W1xcV19dKmNbXFxXX10qayIsICJmdWNrIl0sIFsicFtcXFdfXSp1W1xcV19dKnNbXFxXX10qc1tcXFdfXSp5IiwgInB1c3N5Il0sIFsiW2JwXVtcXFdfXSpbbzBdW1xcV19dKltvMF1bXFxXX10qW2JwXSIsICJib29iIl0sIFsidFtcXFdfXSppW1xcV19dKnRbXFxXX10qcyIsICJ0aXRzIl0sIFsiW2JwXVtvMF1bbzBdW2JwXXM/IiwgImJvb2JzIl0sIFsibltpITFdcHBsW2UzXXM/IiwgIm5pcHBsZSJdLCBbInNbZTNdeFt5dV1hP2w/IiwgInNleHVhbCJdLCBbIltlM11yW28wXXRbaTFdYyIsICJlcm90aWMiXSwgWyJwW28wXXJuW28wXT8iLCAicG9ybiJdLCBbInh4eCsiLCAieHh4Il0sIFsiblthQF1rW2UzXWQiLCAibmFrZWQiXSwgWyJuW3VcXFddZFtlM10iLCAibnVkZSJdLCBbInNbZTNdeCIsICJzZXgiXSwgWyJkW28wXW1baTFdbmF0cltpMV14IiwgImRvbWluYXRyaXgiXV0='
    obfuscation_patterns = json.loads(base64.b64decode(OBFUSCATION_PATTERNS_B64).decode())
    
    # Detect leetspeak/obfuscation patterns
    for pattern, keyword in obfuscation_patterns:
        if re.search(pattern, text_lower):
            logger.warning("NSFW content blocked: obfuscated content detected in prompt")
            return True
    
    # Detect suspicious character substitutions (unicode lookalikes)
    suspicious_chars = {
        'а': 'a', 'е': 'e', 'і': 'i', 'о': 'o', 'р': 'p', 'с': 'c', 'у': 'u', 'х': 'x',
        'ė': 'e', 'ṇ': 'n', 'ū': 'u', 'ṡ': 's', 'ḋ': 'd'
    }
    text_decoded = text_lower
    for char, replacement in suspicious_chars.items():
        text_decoded = text_decoded.replace(char, replacement)
    
    if text_decoded != text_lower:
        # Recheck with decoded text
        for keyword in nsfw_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_decoded):
                logger.warning("NSFW content blocked: unicode-obfuscated content detected in prompt")
                return True
    
    # Context patterns (Base64-encoded)
    CONTEXT_PATTERNS_B64 = 'W1siXFxiKGJlZHxiZWRyb29tfHJvb20pXFxiLipcXGIobmFrZWR8bnVkZXx1bmRyZXNzZWQpXFxiIiwgImJlZHJvb20gbnVkaXR5Il0sIFsiXFxiKHRvdWNoaW5nfHRvdWNofGNhcmVzc2luZ3xjYXJlc3N8cnViYmluZylcXGIuKlxcYihicmVhc3R8Y2hlc3R8Ym9keXxpbnRpbWF0ZSlcXGIoPyEuKihjYW5jZXJ8YXdhcmVuZXNzfGhlYWx0aHxtZWRpY2FsfGV4YW0pKSIsICJpbnRpbWF0ZSB0b3VjaGluZyJdLCBbIlxcYihzcHJlYWR8c3ByZWFkaW5nfG9wZW58b3BlbmluZylcXGIuKlxcYihsZWdzfHRoaWdocylcXGIiLCAic3VnZ2VzdGl2ZSBwb3NlIl0sIFsiXFxiKHdldHxtb2lzdHxkcmlwcGluZylcXGIuKlxcYihib2R5fHNraW58Y2xvdGhlc3xjbG90aGluZylcXGIiLCAic3VnZ2VzdGl2ZSB3ZXRuZXNzIl0sIFsiXFxiKHJlbW92aW5nfHJlbW92ZXx0YWtpbmcgb2ZmfHN0cmlwcGluZylcXGIuKlxcYihjbG90aGVzfGNsb3RoaW5nfGRyZXNzfHNoaXJ0fHBhbnRzKVxcYiIsICJ1bmRyZXNzaW5nIl0sIFsiXFxiKGV4cG9zZWR8cmV2ZWFsaW5nfHNob3dpbmcpXFxiLipcXGIoYnJlYXN0fG5pcHBsZXxnZW5pdGFsKSg/IS4qKGNhbmNlcnxhd2FyZW5lc3N8aGVhbHRofG1lZGljYWx8ZXhhbSkpIiwgImV4cGxpY2l0IGV4cG9zdXJlIl1d'
    context_patterns = json.loads(base64.b64decode(CONTEXT_PATTERNS_B64).decode())
    
    # Context-based detection (combinations of borderline words)
    for pattern, description in context_patterns:
        if re.search(pattern, text_normalized):
            logger.warning("NSFW content blocked: context pattern detected in prompt")
            return True
    
    return False
    
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Use 'Authorization: Bearer <key>' header or session cookie."
    )


async def get_current_user_optional(
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None),
    alice_session: Optional[str] = Cookie(None),
):
    """
    Dependency that returns the current authenticated user (APIKey record) if authenticated.
    
    Returns None if not authenticated (does not raise exception).
    """
    if auth_manager is None:
        return None
    
    # Try session cookie first (contains raw API key)
    if alice_session:
        api_key = auth_manager.verify_api_key(alice_session)
        if api_key:
            return api_key
    
    # Try API key from headers
    key_string = None
    if authorization and authorization.startswith("Bearer "):
        key_string = authorization[7:]
    elif x_api_key:
        key_string = x_api_key
    
    if key_string:
        api_key = auth_manager.verify_api_key(key_string)
        if api_key:
            return api_key
    
    return None


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with OpenAI-compatible error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "type": "api_error",
                "code": str(exc.status_code)
            }
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.exception("Unexpected error: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "type": "internal_error",
                "code": "500"
            }
        }
    )


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns service status, GPU availability, and loaded models.
    """
    if generator is None:
        return HealthResponse(
            status="starting",
            gpu_available=False,
            gpu_stats_available=False,
            models_loaded=0,
            version="1.2.1"
        )
    
    gpu_info = generator.get_gpu_info()
    
    return HealthResponse(
        status="ok",
        gpu_available=gpu_info["gpu_available"],
        gpu_stats_available=gpu_info.get("stats_available", False),
        models_loaded=1 if generator.is_model_loaded else 0,
        version="1.2.1"
    )


@app.get("/v1/generation/defaults")
async def get_generation_defaults():
    """
    Get generation default settings.
    
    Returns default values for steps, guidance, scheduler, dimensions.
    Public endpoint - no auth required.
    """
    return {
        "steps": config.generation.default_steps,
        "guidance_scale": config.generation.default_guidance_scale,
        "scheduler": config.generation.default_scheduler,
        "width": config.generation.default_width,
        "height": config.generation.default_height,
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(access: AccessLevel = Depends(require_access_level(AccessLevel.ANONYMOUS))):
    """
    List available models.
    
    Returns models in OpenAI-compatible format.
    """
    if model_registry is None:
        return ModelsResponse(object="list", data=[])
    
    models = model_registry.list_models()
    
    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id=model.id,
                object="model",
                created=model.created,
                owned_by="alice"
            )
            for model in models
        ]
    )


@app.post("/v1/models/refresh")
async def refresh_models(access: AccessLevel = Depends(require_access_level(AccessLevel.USER))):
    """
    Refresh model registry by rescanning models directory.
    """
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    models = model_registry.refresh()
    loras = model_registry.list_loras()
    logger.info("Model registry refreshed: %d models and %d LoRAs found", len(models), len(loras))
    
    return {
        "status": "ok",
        "models_found": len(models),
        "loras_found": len(loras)
    }


@app.get("/v1/models/{model_id}/info")
async def get_model_info(model_id: str, access: AccessLevel = Depends(require_access_level(AccessLevel.ANONYMOUS))):
    """
    Get detailed information about a specific model.
    
    Returns model type and recommended resolutions for UI adaptation.
    """
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Get model from registry
    model = model_registry.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    
    # Map model types to native resolutions and recommended presets
    resolution_presets = {
        "sd15": {
            "native_resolution": 512,
            "recommended_resolutions": [
                {"width": 512, "height": 512, "label": "Square (512×512)"},
                {"width": 512, "height": 768, "label": "Portrait (512×768)"},
                {"width": 768, "height": 512, "label": "Landscape (768×512)"},
                {"width": 640, "height": 640, "label": "Square (640×640)"},
                {"width": 768, "height": 768, "label": "Large Square (768×768)"}
            ]
        },
        "sd21": {
            "native_resolution": 768,
            "recommended_resolutions": [
                {"width": 768, "height": 768, "label": "Square (768×768)"},
                {"width": 512, "height": 768, "label": "Portrait (512×768)"},
                {"width": 768, "height": 512, "label": "Landscape (768×512)"},
                {"width": 640, "height": 960, "label": "Tall Portrait (640×960)"},
                {"width": 960, "height": 640, "label": "Wide Landscape (960×640)"}
            ]
        },
        "sdxl": {
            "native_resolution": 1024,
            "recommended_resolutions": [
                {"width": 640, "height": 1536, "label": "Ultra Portrait (640×1536)"},
                {"width": 768, "height": 1344, "label": "Tall Portrait (768×1344)"},
                {"width": 832, "height": 1216, "label": "Portrait (832×1216)"},
                {"width": 896, "height": 1152, "label": "Portrait (896×1152)"},
                {"width": 1024, "height": 1024, "label": "Square (1024×1024)"},
                {"width": 1152, "height": 896, "label": "Landscape (1152×896)"},
                {"width": 1216, "height": 832, "label": "Landscape (1216×832)"},
                {"width": 1344, "height": 768, "label": "Wide Landscape (1344×768)"},
                {"width": 1536, "height": 640, "label": "Ultra Wide (1536×640)"}
            ]
        },
        "flux": {
            "native_resolution": 1024,
            "recommended_resolutions": [
                {"width": 1024, "height": 1024, "label": "Square (1024×1024)"},
                {"width": 768, "height": 1344, "label": "Portrait (768×1344)"},
                {"width": 1344, "height": 768, "label": "Landscape (1344×768)"}
            ]
        },
        "sd3": {
            "native_resolution": 1024,
            "recommended_resolutions": [
                {"width": 1024, "height": 1024, "label": "Square (1024×1024)"},
                {"width": 768, "height": 1344, "label": "Portrait (768×1344)"},
                {"width": 1344, "height": 768, "label": "Landscape (1344×768)"}
            ]
        },
        "qwen": {
            "native_resolution": 1328,
            "recommended_resolutions": [
                {"width": 1328, "height": 1328, "label": "Square (1328×1328)"},
                {"width": 928, "height": 1664, "label": "Portrait 9:16 (928×1664)"},
                {"width": 1664, "height": 928, "label": "Landscape 16:9 (1664×928)"},
                {"width": 1104, "height": 1472, "label": "Portrait 3:4 (1104×1472)"},
                {"width": 1472, "height": 1104, "label": "Landscape 4:3 (1472×1104)"},
                {"width": 1056, "height": 1584, "label": "Portrait 2:3 (1056×1584)"},
                {"width": 1584, "height": 1056, "label": "Landscape 3:2 (1584×1056)"}
            ]
        }
    }
    
    # Get presets for this model type, default to sd15 if unknown
    presets = resolution_presets.get(model.model_type, resolution_presets["sd15"])
    
    return {
        "id": model.id,
        "name": model.name,
        "model_type": model.model_type,
        "native_resolution": presets["native_resolution"],
        "recommended_resolutions": presets["recommended_resolutions"],
        "size_mb": model.size_mb,
        "created": model.created
    }


@app.get("/v1/loras", response_model=LoRAsResponse)
async def list_loras(access: AccessLevel = Depends(require_access_level(AccessLevel.ANONYMOUS))):
    """
    List available LoRAs.
    
    Returns LoRAs in a structured format.
    """
    if model_registry is None:
        return LoRAsResponse(object="list", data=[])
    
    loras = model_registry.list_loras()
    
    return LoRAsResponse(
        object="list",
        data=[
            LoRAInfo(
                id=lora.id,
                name=lora.name,
                created=lora.created,
                size_mb=lora.size_mb,
                base_model=lora.base_model
            )
            for lora in loras
        ]
    )


# NOTE: Download cancel must come BEFORE model delete due to route priority
# The /v1/models/{model_id:path} pattern would otherwise match "download/task_id"
@app.delete("/v1/models/download/{task_id}")
async def cancel_download(
    task_id: str,
    access: AccessLevel = Depends(require_access_level(AccessLevel.USER))
):
    """
    Cancel a download task.
    """
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = download_manager.cancel_task(task_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Task not found or already complete: {task_id}")
    
    return {"status": "cancelled", "task_id": task_id}


@app.delete("/v1/models/{model_id:path}")
async def delete_model(
    model_id: str,
    access: AccessLevel = Depends(require_access_level(AccessLevel.ADMIN))
):
    """
    Delete a model. Requires admin privileges.
    
    Args:
        model_id: Model identifier (e.g., sd/model-name)
    """
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Unload model first if it's loaded
    if generator and generator._current_model and generator._current_model == model_registry.get_model_path(model_id):
        generator.unload_model()
    
    success = model_registry.delete_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    return {"status": "deleted", "model_id": model_id}


@app.delete("/v1/loras/{lora_id:path}")
async def delete_lora(
    lora_id: str,
    access: AccessLevel = Depends(require_access_level(AccessLevel.ADMIN))
):
    """
    Delete a LoRA. Requires admin privileges.
    
    Args:
        lora_id: LoRA identifier (e.g., lora/style-lora)
    """
    if model_registry is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = model_registry.delete_lora(lora_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"LoRA not found: {lora_id}")
    
    return {"status": "deleted", "lora_id": lora_id}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    http_request: Request,
    current_user: Any = Depends(get_current_user),
    access: AccessLevel = Depends(require_access_level(AccessLevel.USER))
):
    """
    Generate image via OpenAI-compatible chat completions endpoint.
    
    The prompt is extracted from the last user message.
    Generation parameters come from sam_config.
    
    Supports graceful cancellation when client disconnects.
    
    Returns:
        ChatCompletionResponse with image URL in message content
    """
    if model_registry is None or generator is None or gallery_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    logger.info("Processing chat completion request: %s (user=%s)", request_id, current_user.id)
    
    # Create cancellation token for this request
    cancellation_registry = get_cancellation_registry()
    cancellation_token = cancellation_registry.create_token(request_id)
    
    # Start background task to monitor client connection
    async def monitor_client_disconnect():
        """Monitor for client disconnect and trigger cancellation."""
        try:
            while True:
                if await http_request.is_disconnected():
                    logger.info("Client disconnected, cancelling request: %s", request_id)
                    cancellation_token.cancel()
                    break
                await asyncio.sleep(0.5)  # Check every 500ms
        except Exception as e:
            logger.debug("Disconnect monitor error (expected on completion): %s", e)
    
    # Start monitor task (will be cancelled when request completes)
    monitor_task = asyncio.create_task(monitor_client_disconnect())
    
    try:
        # Extract prompt from messages
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(
                status_code=400,
                detail="No user message found in request"
            )
        
        prompt = user_messages[-1].content
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="Empty prompt in user message"
            )
        
        # Check for NSFW content if blocking is enabled
        if config.server.block_nsfw and check_nsfw_content(prompt):
            raise HTTPException(
                status_code=400,
                detail="NSFW content detected and blocked. This server has NSFW generation disabled."
            )
        
        # Parse A1111-style LoRA syntax from prompt: <lora:name:scale>
        import re
        lora_pattern = r'<lora:([^:>]+):?([0-9.]*)?>'
        prompt_loras = []
        prompt_lora_scales = []
        for match in re.finditer(lora_pattern, prompt, re.IGNORECASE):
            lora_name = match.group(1)
            lora_scale = float(match.group(2)) if match.group(2) else 1.0
            prompt_loras.append(lora_name)
            prompt_lora_scales.append(lora_scale)
            logger.debug("Parsed LoRA from prompt: %s (scale=%.2f)", lora_name, lora_scale)
        
        # Remove LoRA tags from prompt (they shouldn't be sent to the model)
        clean_prompt = re.sub(lora_pattern, '', prompt, flags=re.IGNORECASE).strip()
        # Clean up extra commas/spaces left behind
        clean_prompt = re.sub(r',\s*,', ',', clean_prompt)
        clean_prompt = re.sub(r'\s+', ' ', clean_prompt).strip(' ,')
        if clean_prompt != prompt:
            logger.info("Cleaned prompt (removed LoRA tags): '%s...'", clean_prompt[:50])
            prompt = clean_prompt
        
        # Get model info
        model_info = model_registry.get_model(request.model)
        if not model_info:
            # Try to find by name without prefix
            model_name = request.model.split("/")[-1] if "/" in request.model else request.model
            for model in model_registry.list_models():
                if model.name == model_name:
                    model_info = model
                    break
        
        if not model_info:
            available_models = [m.id for m in model_registry.list_models()]
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {request.model}. Available: {available_models}"
            )
        
        # Extract generation parameters from sam_config
        sam_config = request.sam_config
        
        # Resolve LoRA paths - combine sam_config LoRAs with prompt-parsed LoRAs
        lora_paths = []
        lora_scales = []
        
        # First add LoRAs from sam_config (UI-selected)
        lora_ids = sam_config.lora_paths if sam_config else None
        if lora_ids and len(lora_ids) > 0:
            for lora_id in lora_ids:
                lora_path = model_registry.get_lora_path(lora_id)
                if lora_path:
                    lora_paths.append(lora_path)
                else:
                    logger.warning("LoRA not found: %s (skipping)", lora_id)
            
            config_lora_scales = sam_config.lora_scales if sam_config else None
            if config_lora_scales:
                lora_scales.extend(config_lora_scales)
            else:
                lora_scales.extend([1.0] * len(lora_paths))
        
        # Then add LoRAs parsed from prompt
        if prompt_loras:
            for i, lora_name in enumerate(prompt_loras):
                # Try to find LoRA by name
                lora_path = model_registry.get_lora_path(lora_name)
                if lora_path:
                    lora_paths.append(lora_path)
                    lora_scales.append(prompt_lora_scales[i])
                    logger.info("Resolved LoRA from prompt: %s -> %s (scale=%.2f)", 
                               lora_name, lora_path, prompt_lora_scales[i])
                else:
                    logger.warning("LoRA from prompt not found: %s (skipping)", lora_name)
        
        # Convert to None if empty for generator
        if not lora_paths:
            lora_paths = None
            lora_scales = None
        
        # Debug: Log generation parameters
        gen_scheduler = sam_config.scheduler if sam_config else None
        gen_steps = sam_config.steps if sam_config else None
        gen_width = sam_config.width if sam_config else None
        gen_height = sam_config.height if sam_config else None
        gen_guidance_scale = sam_config.guidance_scale if sam_config else None
        gen_negative_prompt = sam_config.negative_prompt if sam_config else None
        gen_seed = sam_config.seed if sam_config else None
        gen_num_images = sam_config.num_images if sam_config else None
        logger.info("Generation params from sam_config: scheduler=%s, steps=%s, guidance=%.1f, size=%sx%s", 
                    gen_scheduler, gen_steps, gen_guidance_scale if gen_guidance_scale is not None else -1, gen_width, gen_height)
        
        # Check negative_prompt for NSFW content if blocking is enabled
        if config.server.block_nsfw and gen_negative_prompt and check_nsfw_content(gen_negative_prompt):
            raise HTTPException(
                status_code=400,
                detail="NSFW content detected in negative_prompt and blocked. This server has NSFW generation disabled."
            )
        
        # Generate image(s)
        image_paths, metadata = await generator.generate(
            model_path=Path(model_info.path),
            prompt=prompt,
            negative_prompt=gen_negative_prompt or "",
            steps=gen_steps,
            guidance_scale=gen_guidance_scale,
            width=gen_width,
            height=gen_height,
            seed=gen_seed,
            scheduler=gen_scheduler,
            num_images=gen_num_images or 1,
            lora_paths=lora_paths if lora_paths else None,
            lora_scales=lora_scales,
            cancellation_token=cancellation_token,
        )
        
        # Build image URLs and record in gallery
        request_host = http_request.headers.get("host", f"{config.server.host}:{config.server.port}")
        protocol = http_request.headers.get("x-forwarded-proto", "http")
        
        image_urls = []
        for image_path in image_paths:
            image_url = f"{protocol}://{request_host}/images/{image_path.name}"
            image_urls.append(image_url)
            
            # Record each image in gallery
            # If auth is not required, make images public by default
            image_is_public = not config.server.require_auth
            image_id = image_path.stem
            image_record = ImageRecord(
                id=image_id,
                filename=image_path.name,
                owner_api_key_id=current_user.id if current_user else None,
                is_public=image_is_public,
                prompt=prompt,
                negative_prompt=metadata.get("negative_prompt", ""),
                model=model_info.name,
                steps=metadata["steps"],
                guidance_scale=metadata["guidance_scale"],
                width=metadata["width"],
                height=metadata["height"],
                seed=metadata.get("seed"),
                scheduler=metadata["scheduler"],
                generation_time=metadata.get("generation_time"),
                loras=metadata.get("loras"),
                lora_scales=metadata.get("lora_scales"),
            )
            gallery_manager.add_image(image_record)
        
        # Build response content
        num_generated = len(image_urls)
        response_content = f"{num_generated} image{'s' if num_generated > 1 else ''} generated successfully."
        
        # Build metadata
        image_metadata = ImageMetadata(
            prompt=prompt,
            negative_prompt=metadata.get("negative_prompt", ""),
            steps=metadata["steps"],
            guidance_scale=metadata["guidance_scale"],
            seed=metadata.get("seed"),
            model=model_info.name,
            scheduler=metadata["scheduler"],
            width=metadata["width"],
            height=metadata["height"],
        )
        
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ResponseMessage(
                        role="assistant",
                        content=response_content,
                        image_urls=image_urls,
                        metadata=image_metadata
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )
        
    except CancellationError as e:
        # Request was cancelled - return 499 (Client Closed Request)
        logger.info("Request cancelled: %s", request_id)
        raise HTTPException(
            status_code=499,
            detail="Request cancelled by client"
        )
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Image generation failed: {str(e)}"
        )
    finally:
        # Always cleanup: cancel monitor task and unregister token
        if not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        cancellation_registry.unregister(request_id)
        logger.debug("Request cleanup complete: %s", request_id)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get service metrics.
    
    Returns queue depth, GPU utilization, and generation statistics.
    """
    if generator is None:
        return MetricsResponse()
    
    gpu_info = generator.get_gpu_info()
    
    return MetricsResponse(
        queue_depth=generator.get_queue_depth(),  # Pending requests waiting in queue
        active_generations=generator.get_active_generations(),  # Currently executing
        gpu_utilization=gpu_info.get("utilization", 0.0),
        gpu_memory_used=gpu_info.get("memory_used", "0 GB"),
        gpu_memory_total=gpu_info.get("memory_total", "0 GB"),
        models_loaded=1 if generator.is_model_loaded else 0,
        total_generations=generator.total_generations,
        avg_generation_time=generator.get_average_generation_time()
    )


# =============================================================================
# GALLERY ENDPOINTS
# =============================================================================

@app.get("/images/{filename}")
async def serve_image(
    filename: str,
    alice_session: Optional[str] = Cookie(default=None),
    authorization: Optional[str] = Header(default=None)
):
    """
    Serve an image file with access control.
    
    Only the owner, admins, or anyone (for public non-expired images) can access.
    """
    if gallery_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Extract image ID from filename
    image_id = Path(filename).stem
    
    # Get image record
    image_record = gallery_manager.get_image(image_id)
    if not image_record:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Get current user from session or API key
    current_user = None
    is_admin = False
    
    if alice_session and auth_manager:
        # The alice_session cookie contains the raw API key
        api_key = auth_manager.verify_api_key(alice_session)
        if api_key:
            current_user = api_key
            is_admin = api_key.get_access_level() == AccessLevel.ADMIN
    
    if not current_user and authorization and auth_manager:
        # Try API key from header
        token = authorization.replace("Bearer ", "")
        api_key = auth_manager.verify_api_key(token)
        if api_key:
            current_user = api_key
            is_admin = api_key.get_access_level() == AccessLevel.ADMIN
    
    # Check access
    api_key_id = current_user.id if current_user else None
    if not image_record.is_accessible_by(api_key_id, is_admin):
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to access this image"
        )
    
    # Serve the file
    image_path = config.storage.images_directory / filename
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")
    
    return FileResponse(image_path, media_type="image/png")


@app.get("/v1/gallery", response_model=GalleryListResponse)
async def list_gallery_images(
    request: Request,
    limit: int = 100,
    offset: int = 0,
    include_public: bool = True,
    include_private: bool = True,
    current_user: Any = Depends(get_current_user_optional)
):
    """
    List images accessible to the current user.
    
    Returns user's own images (public and private) plus other users' public images.
    Anonymous users only see public images.
    """
    if gallery_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Determine access level
    api_key_id = current_user.id if current_user else None
    is_admin = current_user.get_access_level() == AccessLevel.ADMIN if current_user else False
    
    logger.debug("Gallery list request: api_key_id=%s, is_admin=%s", api_key_id, is_admin)
    
    # List accessible images
    images = gallery_manager.list_images(
        api_key_id=api_key_id,
        is_admin=is_admin,
        include_public=include_public,
        include_private=include_private,
        limit=limit,
        offset=offset
    )
    
    logger.debug("Gallery returning %d images", len(images))
    
    # Build response using request's host header for correct external access
    request_host = request.headers.get("host", f"{config.server.host}:{config.server.port}")
    protocol = request.headers.get("x-forwarded-proto", "http")
    
    data = []
    for img in images:
        is_owner = bool(api_key_id and img.owner_api_key_id == api_key_id)
        
        data.append(GalleryImageInfo(
            id=img.id,
            filename=img.filename,
            url=f"{protocol}://{request_host}/images/{img.filename}",
            thumbnail_url=None,  # TODO: Generate thumbnails
            is_public=img.is_public,
            created_at=img.created_at,
            expires_at=img.expires_at,
            is_owner=is_owner,
            prompt=img.prompt,
            negative_prompt=img.negative_prompt,
            model=img.model,
            steps=img.steps,
            guidance_scale=img.guidance_scale,
            width=img.width,
            height=img.height,
            seed=img.seed,
            scheduler=img.scheduler,
            generation_time=img.generation_time,
            loras=img.loras,
            lora_scales=img.lora_scales,
        ))
    
    return GalleryListResponse(
        data=data,
        total=len(images),  # TODO: Get actual total count
        limit=limit,
        offset=offset
    )


@app.patch("/v1/gallery/{image_id}/privacy")
async def update_image_privacy(
    image_id: str,
    request: UpdateImagePrivacyRequest,
    current_user: Any = Depends(get_current_user)
):
    """
    Update image privacy settings.
    
    Only the owner or admins can update privacy settings.
    """
    if gallery_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Get image
    image = gallery_manager.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Check ownership
    is_admin = current_user.get_access_level() == AccessLevel.ADMIN
    is_owner = image.owner_api_key_id == current_user.id
    
    if not is_owner and not is_admin:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to modify this image"
        )
    
    # Calculate expiration if making public
    expires_at = None
    if request.is_public and request.expires_in_hours:
        expires_at = time.time() + (request.expires_in_hours * 3600)
    elif request.is_public:
        # Use default expiration
        expires_at = time.time() + (config.storage.public_image_expiration_hours * 3600)
    
    # Update privacy
    success = gallery_manager.update_privacy(
        image_id=image_id,
        is_public=request.is_public,
        expires_at=expires_at
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to update privacy")
    
    return {"success": True, "is_public": request.is_public, "expires_at": expires_at}


@app.delete("/v1/gallery/{image_id}")
async def delete_gallery_image(
    image_id: str,
    current_user: Any = Depends(get_current_user)
):
    """
    Delete an image from the gallery.
    
    Only the owner or admins can delete images.
    """
    if gallery_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Get image
    image = gallery_manager.get_image(image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Check ownership
    is_admin = current_user.get_access_level() == AccessLevel.ADMIN
    is_owner = image.owner_api_key_id == current_user.id
    
    if not is_owner and not is_admin:
        raise HTTPException(
            status_code=403,
            detail="You don't have permission to delete this image"
        )
    
    # Delete from gallery
    success = gallery_manager.delete_image(image_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete from gallery")
    
    # Delete image file
    image_path = config.storage.images_directory / image.filename
    if image_path.exists():
        try:
            image_path.unlink()
            logger.info("Deleted image file: %s", image.filename)
        except Exception as e:
            logger.error("Failed to delete image file: %s", e)
            raise HTTPException(status_code=500, detail="Failed to delete image file")
    
    return {"success": True}


@app.get("/v1/gallery/stats", response_model=GalleryStatsResponse)
async def get_gallery_stats(
    access: AccessLevel = Depends(require_access_level(AccessLevel.ADMIN))
):
    """
    Get gallery statistics (admin only).
    """
    if gallery_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    stats = gallery_manager.get_stats()
    return GalleryStatsResponse(**stats)


@app.get("/v1/gallery/config")
async def get_gallery_config():
    """
    Get public gallery configuration (no auth required).
    
    Returns settings like page size that the UI needs.
    """
    return {
        "page_size": config.storage.gallery_page_size
    }


@app.post("/v1/gallery/cleanup")
async def cleanup_expired_images(
    access: AccessLevel = Depends(require_access_level(AccessLevel.ADMIN))
):
    """
    Clean up expired public images (admin only).
    """
    if gallery_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    cleaned = gallery_manager.cleanup_expired(config.storage.images_directory)
    return {"success": True, "cleaned": cleaned}


# =============================================================================
# DOWNLOAD ENDPOINTS
# =============================================================================

@app.post("/v1/models/search/civitai")
async def search_civitai(
    request: CivitAISearchRequest,
    access: AccessLevel = Depends(require_access_level(AccessLevel.USER))
):
    """
    Search CivitAI for models.
    
    Uses local cache if available, otherwise queries CivitAI API directly.
    Returns matching models with version and file information.
    """
    # Use cache if available
    logger.info("search_civitai: model_cache_service=%s, enabled=%s", model_cache_service is not None, config.model_cache.enabled)
    if model_cache_service and config.model_cache.enabled:
        logger.info("Using model cache for CivitAI search: query=%s, types=%s, nsfw=%s, limit=%s", request.query, request.types, request.nsfw, request.limit)
        models, total = model_cache_service.search(
            source="civitai",
            query=request.query,
            types=request.types,
            nsfw=request.nsfw,
            sort="rating" if request.sort == "Highest Rated" else "download_count",
            limit=request.limit,
            offset=(request.page - 1) * request.limit if request.page > 1 else 0,
        )
        
        logger.info("Cache search returned %d models (total=%d)", len(models), total)
        return {
            "object": "list",
            "data": models,
            "total": total,
            "page": request.page,
            "cached": True,
        }
    
    # Fallback to live API search
    logger.info("Falling back to live CivitAI API search")
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    models = await download_manager.search_civitai(
        query=request.query,
        types=request.types,
        sort=request.sort,
        nsfw=request.nsfw,
        limit=request.limit,
        page=request.page,
    )
    
    return {
        "object": "list",
        "data": [m.to_dict() for m in models],
        "cached": False,
    }


@app.post("/v1/models/search/huggingface")
async def search_huggingface(
    request: HuggingFaceSearchRequest,
    access: AccessLevel = Depends(require_access_level(AccessLevel.USER))
):
    """
    Search HuggingFace for diffusion models.
    
    Uses local cache if available, otherwise queries HuggingFace API directly.
    Returns matching models.
    """
    # Use cache if available
    if model_cache_service and config.model_cache.enabled:
        models, total = model_cache_service.search(
            source="huggingface",
            query=request.query,
            sort="download_count",
            limit=request.limit,
            offset=0,
        )
        
        return {
            "object": "list",
            "data": models,
            "total": total,
            "cached": True,
        }
    
    # Fallback to live API search
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    models = await download_manager.search_huggingface(
        query=request.query,
        filter_tags=request.filter_tags,
        sort=request.sort,
        limit=request.limit,
    )
    
    return {
        "object": "list",
        "data": [m.to_dict() for m in models],
        "cached": False,
    }


@app.post("/v1/models/download/civitai")
async def download_civitai(
    request: CivitAIDownloadRequest,
    access: AccessLevel = Depends(require_access_level(AccessLevel.USER))
):
    """
    Queue download of a CivitAI model.
    
    Returns the download task ID.
    """
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    task_id = await download_manager.download_civitai(
        model_id=request.model_id,
        version_id=request.version_id,
        file_id=request.file_id,
        model_type=request.model_type,
    )
    
    if not task_id:
        raise HTTPException(status_code=400, detail="Failed to queue download")
    
    task = download_manager.get_task(task_id)
    return {
        "status": "queued",
        "task_id": task_id,
        "task": task.to_dict() if task else None
    }


@app.get("/v1/models/cache/status")
async def get_cache_status(
    access: AccessLevel = Depends(require_access_level(AccessLevel.USER))
):
    """
    Get model cache synchronization status.
    
    Returns last sync times, record counts, and current sync progress.
    """
    if not model_cache_service or not config.model_cache.enabled:
        return {
            "enabled": False,
            "message": "Model cache is disabled"
        }
    
    status = model_cache_service.get_sync_status()
    
    return {
        "enabled": True,
        **status.to_dict()
    }


@app.post("/v1/models/download/huggingface")
async def download_huggingface(
    request: HuggingFaceDownloadRequest,
    access: AccessLevel = Depends(require_access_level(AccessLevel.USER))
):
    """
    Queue download of a HuggingFace model.
    
    Returns the download task ID.
    """
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    task_id = await download_manager.download_huggingface(
        model_id=request.model_id,
        filename=request.filename,
        revision=request.revision,
    )
    
    if not task_id:
        raise HTTPException(status_code=400, detail="Failed to queue download")
    
    task = download_manager.get_task(task_id)
    return {
        "status": "queued",
        "task_id": task_id,
        "task": task.to_dict() if task else None
    }


@app.post("/v1/models/download/url")
async def download_url(
    request: DirectDownloadRequest,
    access: AccessLevel = Depends(require_access_level(AccessLevel.USER))
):
    """
    Queue download from a direct URL.
    
    Returns the download task ID.
    """
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    task_id = await download_manager.download_url(
        url=request.url,
        filename=request.filename,
        is_lora=request.is_lora,
    )
    
    if not task_id:
        raise HTTPException(status_code=400, detail="Failed to queue download")
    
    task = download_manager.get_task(task_id)
    return {
        "status": "queued",
        "task_id": task_id,
        "task": task.to_dict() if task else None
    }


@app.get("/v1/models/download/status", response_model=DownloadListResponse)
async def list_downloads(access: AccessLevel = Depends(require_access_level(AccessLevel.ANONYMOUS))):
    """
    List all download tasks.
    """
    if download_manager is None:
        return DownloadListResponse(object="list", data=[])
    
    tasks = download_manager.list_tasks()
    
    return DownloadListResponse(
        object="list",
        data=[
            DownloadTaskInfo(
                id=t.id,
                source=t.source.value,
                name=t.name,
                status=t.status.value,
                progress=t.progress,
                total_size=t.total_size,
                downloaded_size=t.downloaded_size,
                speed=t.speed,
                error=t.error,
            )
            for t in tasks
        ]
    )


@app.get("/v1/models/download/status/{task_id}")
async def get_download_status(
    task_id: str,
    access: AccessLevel = Depends(require_access_level(AccessLevel.ANONYMOUS))
):
    """
    Get status of a specific download task.
    """
    if download_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    task = download_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    
    return task.to_dict()


# =============================================================================
# AUTHENTICATION ENDPOINTS
# =============================================================================

async def verify_admin_key(x_admin_key: Optional[str] = Header(None)) -> bool:
    """Verify admin API key."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    if not x_admin_key:
        raise HTTPException(
            status_code=401,
            detail="Admin key required. Use 'X-Admin-Key' header."
        )
    
    api_key = auth_manager.verify_api_key(x_admin_key)
    if not api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")
    
    if not api_key.is_admin:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    return True


@app.post("/v1/auth/keys/generate", response_model=APIKeyCreatedResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    admin: bool = Depends(verify_admin_key)
):
    """
    Create a new API key. Requires admin privileges.
    
    The key is only returned once - store it securely!
    
    Access levels:
    - anonymous: Can view models, health checks only
    - user: Can generate images, download models, view all
    - admin: Full control including key management
    """
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    plaintext_key, api_key = auth_manager.create_api_key(
        name=request.name,
        is_admin=request.is_admin,
        access_level=request.access_level,
        rate_limit=request.rate_limit,
    )
    
    return APIKeyCreatedResponse(
        key=plaintext_key,
        info=APIKeyInfo(
            id=api_key.id,
            name=api_key.name,
            created_at=api_key.created_at,
            last_used=api_key.last_used,
            is_admin=api_key.is_admin,
            access_level=api_key.access_level,
            rate_limit=api_key.rate_limit,
            enabled=api_key.enabled,
        )
    )


@app.get("/v1/auth/keys", response_model=APIKeyListResponse)
async def list_api_keys(admin: bool = Depends(verify_admin_key)):
    """List all API keys. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    keys = auth_manager.list_api_keys()
    
    return APIKeyListResponse(
        object="list",
        data=[
            APIKeyInfo(
                id=k.id,
                name=k.name,
                created_at=k.created_at,
                last_used=k.last_used,
                is_admin=k.is_admin,
                access_level=k.access_level,
                rate_limit=k.rate_limit,
                enabled=k.enabled,
            )
            for k in keys
        ]
    )


@app.delete("/v1/auth/keys/{key_id}")
async def delete_api_key(key_id: str, admin: bool = Depends(verify_admin_key)):
    """Delete an API key. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = auth_manager.delete_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Key not found: {key_id}")
    
    return {"status": "deleted", "key_id": key_id}


@app.post("/v1/auth/keys/{key_id}/revoke")
async def revoke_api_key(key_id: str, admin: bool = Depends(verify_admin_key)):
    """Revoke (disable) an API key. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = auth_manager.revoke_api_key(key_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Key not found: {key_id}")
    
    return {"status": "revoked", "key_id": key_id}


@app.post("/v1/auth/sessions", response_model=SessionCreatedResponse)
async def create_session(access: AccessLevel = Depends(require_access_level(AccessLevel.USER))):
    """Create a new session."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    session = auth_manager.create_session()
    
    return SessionCreatedResponse(
        session_id=session.id,
        token=session.token,
        expires_at=session.expires_at,
    )


@app.get("/v1/auth/sessions", response_model=SessionListResponse)
async def list_sessions(admin: bool = Depends(verify_admin_key)):
    """List all active sessions. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    sessions = auth_manager.list_sessions()
    
    return SessionListResponse(
        object="list",
        data=[
            SessionInfo(
                id=s.id,
                api_key_id=s.api_key_id,
                created_at=s.created_at,
                last_accessed=s.last_accessed,
                expires_at=s.expires_at,
                metadata=s.metadata,
            )
            for s in sessions
        ]
    )


@app.delete("/v1/auth/sessions/{session_id}")
async def delete_session(session_id: str, admin: bool = Depends(verify_admin_key)):
    """Delete a session. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = auth_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    return {"status": "deleted", "session_id": session_id}


@app.get("/v1/auth/me")
async def get_current_user(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
):
    """
    Get information about the current authenticated user.
    
    Returns access level, admin status, and session timeout info.
    """
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Extract API key
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    elif x_api_key:
        api_key = x_api_key
    
    if not api_key:
        return {
            "authenticated": False,
            "access_level": "anonymous",
            "is_admin": False,
            "session_timeout_seconds": SESSION_TIMEOUT_SECONDS,
            "inactivity_timeout_seconds": SESSION_INACTIVITY_TIMEOUT_SECONDS,
        }
    
    # Check against registered keys
    key_record = auth_manager.verify_api_key(api_key)
    if key_record:
        return {
            "authenticated": True,
            "access_level": key_record.access_level,
            "is_admin": key_record.is_admin,
            "key_id": key_record.id,
            "name": key_record.name,
            "session_timeout_seconds": SESSION_TIMEOUT_SECONDS,
            "inactivity_timeout_seconds": SESSION_INACTIVITY_TIMEOUT_SECONDS,
        }
    
    return {
        "authenticated": False,
        "access_level": "anonymous",
        "is_admin": False,
        "session_timeout_seconds": SESSION_TIMEOUT_SECONDS,
        "inactivity_timeout_seconds": SESSION_INACTIVITY_TIMEOUT_SECONDS,
    }


@app.get("/v1/auth/stats", response_model=AuthStatsResponse)
async def get_auth_stats(admin: bool = Depends(verify_admin_key)):
    """Get authentication statistics. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    stats = auth_manager.get_stats()
    
    return AuthStatsResponse(
        total_keys=stats["total_keys"],
        active_keys=stats["active_keys"],
        admin_keys=stats["admin_keys"],
        total_sessions=stats["total_sessions"],
        active_sessions=stats["active_sessions"],
    )


@app.post("/v1/auth/bootstrap")
async def bootstrap_admin():
    """
    Bootstrap the first admin key.
    
    This endpoint only works if no API keys exist yet.
    Use this to create the initial admin key.
    """
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Check if any keys exist
    existing_keys = auth_manager.list_api_keys()
    if len(existing_keys) > 0:
        raise HTTPException(
            status_code=403,
            detail="Bootstrap not allowed - API keys already exist. Use admin key to create new keys."
        )
    
    # Create first admin key
    plaintext_key, api_key = auth_manager.create_api_key(
        name="Initial Admin Key",
        is_admin=True,
        rate_limit=1000,
    )
    
    logger.info("Bootstrap: Created initial admin key")
    
    return {
        "message": "Admin key created. Store this key securely - it will not be shown again!",
        "key": plaintext_key,
        "key_id": api_key.id,
    }


# =============================================================================
# USER REGISTRATION ENDPOINTS
# =============================================================================

@app.get("/v1/auth/registration-mode")
async def get_registration_mode():
    """Get the current registration mode. Public endpoint."""
    return {
        "mode": config.server.registration_mode,
        "require_auth": config.server.require_auth,
    }


@app.post("/v1/auth/register")
async def register_user(
    name: str,
    invite_code: Optional[str] = None,
):
    """
    Register a new user account.
    
    Behavior depends on registration_mode config:
    - open: Creates user key immediately
    - invite: Requires valid invite code
    - approval: Creates pending registration for admin approval
    - disabled: Returns error
    """
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    mode = config.server.registration_mode
    
    if mode == "disabled":
        raise HTTPException(
            status_code=403,
            detail="User registration is disabled. Contact an administrator."
        )
    
    if mode == "open":
        # Create key immediately
        plaintext_key, api_key = auth_manager.create_api_key(
            name=name,
            is_admin=False,
            access_level="user",
        )
        logger.info("Open registration: Created key for %s", name)
        return {
            "status": "created",
            "message": "Account created successfully. Store your API key securely!",
            "key": plaintext_key,
            "key_id": api_key.id,
        }
    
    elif mode == "invite":
        if not invite_code:
            raise HTTPException(
                status_code=400,
                detail="Invite code required for registration."
            )
        
        invite = auth_manager.verify_invite(invite_code)
        if not invite:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired invite code."
            )
        
        # Create key and use invite
        plaintext_key, api_key = auth_manager.create_api_key(
            name=name,
            is_admin=False,
            access_level="user",
        )
        auth_manager.use_invite(invite_code, api_key.id)
        
        logger.info("Invite registration: Created key for %s (invite: %s)", name, invite_code[:8])
        return {
            "status": "created",
            "message": "Account created successfully. Store your API key securely!",
            "key": plaintext_key,
            "key_id": api_key.id,
        }
    
    elif mode == "approval":
        # Check for duplicate pending registration
        existing = [p for p in auth_manager.list_pending_registrations("pending") 
                   if p.name.lower() == name.lower()]
        if existing:
            raise HTTPException(
                status_code=400,
                detail="A registration request with this name already exists."
            )
        
        # Create pending registration
        pending = auth_manager.create_pending_registration(name)
        
        logger.info("Approval registration: Created pending for %s (%s)", name, pending.id)
        return {
            "status": "pending",
            "message": "Registration request submitted. An administrator will review your request.",
            "registration_id": pending.id,
        }
    
    raise HTTPException(status_code=500, detail="Invalid registration mode")


# =============================================================================
# INVITE CODE MANAGEMENT (Admin)
# =============================================================================

@app.post("/v1/auth/invites")
async def create_invite(
    request: CreateInviteRequest,
    admin: bool = Depends(verify_admin_key),
    x_admin_key: Optional[str] = Header(None),
):
    """Create a new invite code. Requires admin privileges.
    
    Set uses=0 for unlimited uses.
    """
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Get admin key ID for tracking
    admin_key = auth_manager.verify_api_key(x_admin_key) if x_admin_key else None
    created_by = admin_key.id if admin_key else "unknown"
    
    invite = auth_manager.create_invite(
        created_by=created_by,
        uses=request.uses,
        expires_hours=request.expires_hours,
    )
    
    return {
        "code": invite.code,
        "uses_remaining": invite.uses_remaining,
        "expires_at": invite.expires_at,
        "created_at": invite.created_at,
    }


@app.get("/v1/auth/invites")
async def list_invites(admin: bool = Depends(verify_admin_key)):
    """List all invite codes. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    invites = auth_manager.list_invites()
    
    return {
        "data": [
            {
                "code": i.code,
                "created_by": i.created_by,
                "created_at": i.created_at,
                "expires_at": i.expires_at,
                "uses_remaining": i.uses_remaining,
                "used_by": i.used_by,
                "valid": i.is_valid(),
            }
            for i in invites
        ]
    }


@app.delete("/v1/auth/invites/{code}")
async def delete_invite(code: str, admin: bool = Depends(verify_admin_key)):
    """Delete an invite code. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = auth_manager.delete_invite(code)
    if not success:
        raise HTTPException(status_code=404, detail=f"Invite not found: {code}")
    
    return {"status": "deleted", "code": code}


# =============================================================================
# PENDING REGISTRATION MANAGEMENT (Admin)
# =============================================================================

@app.get("/v1/auth/pending")
async def list_pending_registrations(
    status: Optional[str] = None,
    admin: bool = Depends(verify_admin_key),
):
    """List pending registrations. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    pending = auth_manager.list_pending_registrations(status)
    
    return {
        "data": [
            {
                "id": p.id,
                "name": p.name,
                "status": p.status,
                "created_at": p.created_at,
                "approved_by": p.approved_by,
                "approved_at": p.approved_at,
                "api_key_id": p.api_key_id,
            }
            for p in pending
        ]
    }


@app.post("/v1/auth/pending/{reg_id}/approve")
async def approve_registration(
    reg_id: str,
    admin: bool = Depends(verify_admin_key),
    x_admin_key: Optional[str] = Header(None),
):
    """Approve a pending registration. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    admin_key = auth_manager.verify_api_key(x_admin_key) if x_admin_key else None
    approved_by = admin_key.id if admin_key else "unknown"
    
    result = auth_manager.approve_registration(reg_id, approved_by)
    if not result:
        raise HTTPException(status_code=404, detail=f"Pending registration not found: {reg_id}")
    
    plaintext_key, api_key = result
    
    return {
        "status": "approved",
        "key": plaintext_key,
        "key_id": api_key.id,
        "message": "Registration approved. Provide this API key to the user.",
    }


@app.post("/v1/auth/pending/{reg_id}/reject")
async def reject_registration(
    reg_id: str,
    admin: bool = Depends(verify_admin_key),
    x_admin_key: Optional[str] = Header(None),
):
    """Reject a pending registration. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    admin_key = auth_manager.verify_api_key(x_admin_key) if x_admin_key else None
    rejected_by = admin_key.id if admin_key else "unknown"
    
    success = auth_manager.reject_registration(reg_id, rejected_by)
    if not success:
        raise HTTPException(status_code=404, detail=f"Pending registration not found: {reg_id}")
    
    return {"status": "rejected", "registration_id": reg_id}


@app.delete("/v1/auth/pending/{reg_id}")
async def delete_pending_registration(
    reg_id: str,
    admin: bool = Depends(verify_admin_key),
):
    """Delete a pending registration. Requires admin privileges."""
    if auth_manager is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    success = auth_manager.delete_pending_registration(reg_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Pending registration not found: {reg_id}")
    
    return {"status": "deleted", "registration_id": reg_id}


# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@app.get("/v1/admin/config")
async def get_config(admin: bool = Depends(verify_admin_key)):
    """
    Get current configuration. Requires admin privileges.
    
    Returns the current config as a dictionary (passwords/keys redacted).
    """
    return {
        "server": {
            "host": config.server.host,
            "port": config.server.port,
            "api_key": "***" if config.server.api_key else None,
            "require_auth": config.server.require_auth,
            "registrationMode": config.server.registration_mode,
            "session_timeout_seconds": config.server.session_timeout_seconds,
            "block_nsfw": config.server.block_nsfw,
        },
        "models": {
            "directory": str(config.models.directory),
            "auto_unload_timeout": config.models.auto_unload_timeout,
            "default_model": config.models.default_model,
            "civitai_api_key": "***" if config.models.civitai_api_key else None,
            "huggingface_token": "***" if config.models.huggingface_token else None,
        },
        "generation": {
            "default_steps": config.generation.default_steps,
            "default_guidance_scale": config.generation.default_guidance_scale,
            "default_scheduler": config.generation.default_scheduler,
            "default_width": config.generation.default_width,
            "default_height": config.generation.default_height,
            "max_concurrent": config.generation.max_concurrent,
            "request_timeout": config.generation.request_timeout,
            "backend": config.generation.backend,
            "sdcpp_binary": str(config.generation.sdcpp_binary) if config.generation.sdcpp_binary else None,
            "sdcpp_threads": config.generation.sdcpp_threads,
            "force_cpu": config.generation.force_cpu,
            "force_float32": config.generation.force_float32,
            "force_bfloat16": config.generation.force_bfloat16,
            "device_map": config.generation.device_map or "",
            "enable_vae_slicing": config.generation.enable_vae_slicing,
            "enable_vae_tiling": config.generation.enable_vae_tiling,
            "enable_model_cpu_offload": config.generation.enable_model_cpu_offload,
            "enable_sequential_cpu_offload": config.generation.enable_sequential_cpu_offload,
            "attention_slice_size": config.generation.attention_slice_size,
            "vae_decode_cpu": config.generation.vae_decode_cpu,
        },
        "storage": {
            "images_directory": str(config.storage.images_directory),
            "max_storage_gb": config.storage.max_storage_gb,
            "retention_days": config.storage.retention_days,
            "gallery_page_size": config.storage.gallery_page_size,
        },
        "logging": {
            "level": config.logging.level,
            "file": str(config.logging.file),
            "max_size_mb": config.logging.max_size_mb,
            "backup_count": config.logging.backup_count,
        },
    }


@app.put("/v1/admin/config")
async def update_config(
    updates: Dict[str, Any],
    admin: bool = Depends(verify_admin_key)
):
    """
    Update configuration. Requires admin privileges.
    
    Accepts a dictionary of section.key = value pairs to update.
    Changes are saved to the config file and take effect on next restart.
    
    Example:
        {
            "server.require_auth": true,
            "generation.default_steps": 30,
            "models.civitai_api_key": "your-key-here"
        }
    """
    import yaml
    
    # Find config file path - check ALICE_CONFIG env var first, then known paths
    env_config = os.environ.get("ALICE_CONFIG")
    
    config_paths = [
        Path(env_config) if env_config else None,
        Path.home() / ".config" / "alice" / "config.yaml",
        Path.home() / ".config" / "ALICE" / "config.yaml",  # Legacy path
        Path("/etc/alice/config.yaml"),
        Path("config.yaml"),
    ]
    
    config_file = None
    for path in config_paths:
        if path and path.exists():
            config_file = path
            break
    
    if not config_file:
        raise HTTPException(status_code=500, detail="Config file not found")
    
    # Read current config
    with open(config_file) as f:
        current_config = yaml.safe_load(f) or {}
    
    # Apply updates
    changes_made = []
    for key, value in updates.items():
        parts = key.split(".")
        if len(parts) != 2:
            continue
        
        section, setting = parts
        
        # Ensure section exists
        if section not in current_config:
            current_config[section] = {}
        
        # Update value
        old_value = current_config[section].get(setting)
        current_config[section][setting] = value
        changes_made.append({
            "key": key,
            "old_value": "***" if "key" in setting.lower() or "token" in setting.lower() else old_value,
            "new_value": "***" if "key" in setting.lower() or "token" in setting.lower() else value,
        })
        
        logger.info("Config updated: %s = %s", key, "***" if "key" in setting.lower() or "token" in setting.lower() else value)
    
    # Write updated config
    try:
        with open(config_file, "w") as f:
            yaml.dump(current_config, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")
    
    return {
        "status": "updated",
        "message": "Configuration saved. Restart service for changes to take effect.",
        "changes": changes_made,
        "config_file": str(config_file),
    }


@app.post("/v1/admin/cache/sync")
async def trigger_cache_sync(
    source: Optional[str] = None,
    admin: bool = Depends(verify_admin_key)
):
    """
    Manually trigger model catalog synchronization.
    Requires admin privileges.
    
    Args:
        source: Optional source to sync ('civitai', 'huggingface', or null for both)
    
    Returns sync status.
    """
    if not model_cache_service or not config.model_cache.enabled:
        raise HTTPException(
            status_code=503,
            detail="Model cache is disabled"
        )
    
    # Check if sync is already running
    status = model_cache_service.get_sync_status()
    if status.sync_in_progress:
        raise HTTPException(
            status_code=409,
            detail="Sync already in progress"
        )
    
    # Start sync as background task
    async def sync_task():
        try:
            if source is None or source == "civitai":
                logger.info("Admin triggered CivitAI sync")
                await model_cache_service.sync_civitai()
            
            if source is None or source == "huggingface":
                logger.info("Admin triggered HuggingFace sync")
                await model_cache_service.sync_huggingface()
            
            logger.info("Manual sync complete")
        except Exception as e:
            logger.error("Manual sync failed: %s", e)
    
    asyncio.create_task(sync_task())
    
    return {
        "status": "started",
        "message": f"Sync started for {source or 'all sources'}",
        "source": source or "all"
    }


@app.post("/v1/admin/restart")
async def restart_service(
    admin: bool = Depends(verify_admin_key)
):
    """
    Restart the ALICE service. Requires admin privileges.
    
    This endpoint triggers a graceful restart of the service.
    Works when running as a systemd user service.
    """
    import subprocess
    import os
    import signal
    
    # Try systemctl restart first (for systemd service)
    try:
        # Check if we're running as a systemd service
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "alice"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            # We're running as a systemd service, use systemctl to restart
            subprocess.Popen(
                ["systemctl", "--user", "restart", "alice"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return {
                "status": "restarting",
                "message": "Service restart initiated via systemctl. The service will restart momentarily.",
                "method": "systemctl"
            }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback: Send SIGTERM to self (graceful shutdown)
    # The service manager or process supervisor should restart us
    logger.info("Initiating graceful shutdown for restart...")
    
    # Schedule the shutdown after response is sent
    import asyncio
    async def delayed_shutdown():
        await asyncio.sleep(0.5)  # Give time for response to be sent
        os.kill(os.getpid(), signal.SIGTERM)
    
    asyncio.create_task(delayed_shutdown())
    
    return {
        "status": "restarting",
        "message": "Service shutdown initiated. If running under a process supervisor, it will restart automatically.",
        "method": "sigterm"
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        log_level=config.logging.level.lower(),
    )
