# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Pydantic Schemas

OpenAI-compatible request/response models for the API.
These MUST match the formats expected by SAM clients.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# REQUEST MODELS
# =============================================================================

class ChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: str = Field(..., description="Message role: system, user, assistant")
    content: Optional[str] = Field(None, description="Message content (prompt)")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, alias="toolCalls")
    tool_call_id: Optional[str] = Field(None, alias="toolCallId")

    model_config = ConfigDict(populate_by_name=True)


class SAMImageConfig(BaseModel):
    """
    SAM-specific image generation configuration.
    
    Passed in the sam_config field of chat completion requests.
    All fields default to None so server defaults from config.yaml are used.
    
    NOTE: Width and height will be automatically rounded to the nearest multiple of 8
    if not already divisible by 8 (required for Stable Diffusion VAE).
    
    For image-to-image (img2img) generation:
    - Set input_images to a list of base64-encoded images
    - Or set input_image_urls to a list of URLs to fetch images from
    - The model must support img2img (e.g., Qwen-Image-Edit-2511)
    """
    negative_prompt: Optional[str] = Field(default=None, description="Negative prompt")
    steps: Optional[int] = Field(default=None, ge=1, le=150, description="Inference steps")
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=20.0, description="Guidance scale (0 for SD Turbo)")
    width: Optional[int] = Field(default=None, ge=64, le=2048, description="Image width (will be rounded to multiple of 8)")
    height: Optional[int] = Field(default=None, ge=64, le=2048, description="Image height (will be rounded to multiple of 8)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    scheduler: Optional[str] = Field(default=None, description="Scheduler name")
    num_images: Optional[int] = Field(default=None, ge=1, le=100, description="Number of images")
    lora_paths: Optional[List[str]] = Field(default=None, description="List of LoRA IDs to apply")
    lora_scales: Optional[List[float]] = Field(default=None, description="LoRA weights (0.0-1.0)")
    input_images: Optional[List[str]] = Field(default=None, description="Base64-encoded input images for img2img")
    input_image_urls: Optional[List[str]] = Field(default=None, description="URLs of input images for img2img")
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="img2img denoising strength (0.0=no change, 1.0=full denoise)")


class ChatCompletionRequest(BaseModel):
    """
    OpenAI-compatible chat completion request.
    
    For image generation, the prompt is extracted from the last user message.
    Generation parameters come from sam_config.
    """
    model: str = Field(..., description="Model ID (e.g., sd/stable-diffusion-v1-5)")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    temperature: Optional[float] = Field(default=0.7, description="Sampling temperature (unused for SD)")
    top_p: Optional[float] = Field(default=None, alias="topP")
    max_tokens: Optional[int] = Field(default=None, alias="maxTokens")
    stream: Optional[bool] = Field(default=False, description="Stream response (not supported)")
    sam_config: Optional[SAMImageConfig] = Field(default=None, alias="samConfig", description="Image generation config")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class Usage(BaseModel):
    """Token usage statistics (always 0 for image generation)."""
    prompt_tokens: int = Field(default=0, alias="promptTokens")
    completion_tokens: int = Field(default=0, alias="completionTokens")
    total_tokens: int = Field(default=0, alias="totalTokens")

    model_config = ConfigDict(populate_by_name=True)


class ImageMetadata(BaseModel):
    """Metadata about generated image."""
    prompt: str
    negative_prompt: Optional[str] = None
    steps: int
    guidance_scale: float
    seed: Optional[int] = None
    model: str
    scheduler: str
    width: int
    height: int
    mode: str = Field(default="txt2img", description="Generation mode: txt2img or img2img")
    input_image_count: Optional[int] = Field(default=None, description="Number of input images (img2img only)")


class ResponseMessage(BaseModel):
    """
    Response message with optional image data.
    
    For image generation, content contains a description and
    image_urls contains the generated image URLs.
    """
    role: str = Field(default="assistant")
    content: str = Field(..., description="Response text")
    image_urls: Optional[List[str]] = Field(default=None, description="Generated image URLs")
    metadata: Optional[ImageMetadata] = Field(default=None, description="Generation metadata")


class ChatCompletionChoice(BaseModel):
    """Single choice in chat completion response."""
    index: int = Field(default=0)
    message: ResponseMessage
    finish_reason: str = Field(default="stop", alias="finishReason")

    model_config = ConfigDict(populate_by_name=True)


class ChatCompletionResponse(BaseModel):
    """
    OpenAI-compatible chat completion response.
    
    This format MUST match ServerOpenAIChatResponse in SAM.
    """
    id: str = Field(..., description="Unique response ID")
    object: str = Field(default="chat.completion")
    created: int = Field(..., description="Unix timestamp")
    model: str = Field(..., description="Model ID used")
    choices: List[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# MODEL LIST MODELS
# =============================================================================

class ModelInfo(BaseModel):
    """OpenAI model information format."""
    id: str = Field(..., description="Model ID (e.g., sd/stable-diffusion-v1-5)")
    object: str = Field(default="model")
    created: int = Field(..., description="Unix timestamp")
    owned_by: str = Field(default="alice", alias="ownedBy")

    model_config = ConfigDict(populate_by_name=True)


class LoRAInfo(BaseModel):
    """LoRA information format."""
    id: str = Field(..., description="LoRA ID (e.g., lora/style-lora)")
    name: str = Field(..., description="LoRA name")
    created: int = Field(..., description="Unix timestamp")
    size_mb: int = Field(default=0, alias="sizeMb", description="Size in MB")
    base_model: Optional[str] = Field(default=None, alias="baseModel", description="Compatible base model type")

    model_config = ConfigDict(populate_by_name=True)


class LoRAsResponse(BaseModel):
    """LoRAs list response."""
    object: str = Field(default="list")
    data: List[LoRAInfo]


class ModelsResponse(BaseModel):
    """OpenAI models list response."""
    object: str = Field(default="list")
    data: List[ModelInfo]


# =============================================================================
# HEALTH & METRICS MODELS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="ok")
    gpu_available: bool = Field(default=False, alias="gpuAvailable")
    gpu_stats_available: bool = Field(default=False, alias="gpuStatsAvailable")
    models_loaded: int = Field(default=0, alias="modelsLoaded")
    version: str = Field(default="1.0.0")

    model_config = ConfigDict(populate_by_name=True)


class MetricsResponse(BaseModel):
    """Service metrics response."""
    queue_depth: int = Field(default=0, alias="queueDepth")
    active_generations: int = Field(default=0, alias="activeGenerations")
    gpu_utilization: float = Field(default=0.0, alias="gpuUtilization")
    gpu_memory_used: str = Field(default="0 GB", alias="gpuMemoryUsed")
    gpu_memory_total: str = Field(default="0 GB", alias="gpuMemoryTotal")
    models_loaded: int = Field(default=0, alias="modelsLoaded")
    total_generations: int = Field(default=0, alias="totalGenerations")
    avg_generation_time: float = Field(default=0.0, alias="avgGenerationTime")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# ERROR MODELS
# =============================================================================

class ErrorDetail(BaseModel):
    """Error detail in response."""
    message: str
    type: str = Field(default="api_error")
    code: str


class ErrorResponse(BaseModel):
    """API error response format."""
    error: ErrorDetail


# =============================================================================
# DOWNLOAD MODELS
# =============================================================================

class CivitAISearchRequest(BaseModel):
    """Request to search CivitAI."""
    query: str = Field(default="", description="Search query (empty = browse mode)")
    types: Optional[List[str]] = Field(default=None, description="Model types (Checkpoint, LORA)")
    sort: str = Field(default="Highest Rated", description="Sort order")
    nsfw: bool = Field(default=True, description="Include NSFW (admins only)")
    limit: int = Field(default=100, ge=1, le=200, description="Max results")
    page: int = Field(default=1, ge=1, description="Page number (browse mode only)")


class CivitAIDownloadRequest(BaseModel):
    """Request to download from CivitAI."""
    model_id: int = Field(..., alias="modelId", description="CivitAI model ID")
    version_id: Optional[int] = Field(default=None, alias="versionId", description="Version ID")
    file_id: Optional[int] = Field(default=None, alias="fileId", description="File ID")
    model_type: str = Field(default="Checkpoint", alias="modelType", description="Model type")

    model_config = ConfigDict(populate_by_name=True)


class HuggingFaceSearchRequest(BaseModel):
    """Request to search HuggingFace."""
    query: str = Field(..., description="Search query")
    filter_tags: Optional[List[str]] = Field(default=None, alias="filterTags", description="Filter tags")
    sort: str = Field(default="downloads", description="Sort field")
    limit: int = Field(default=20, ge=1, le=100, description="Max results")

    model_config = ConfigDict(populate_by_name=True)


class HuggingFaceDownloadRequest(BaseModel):
    """Request to download from HuggingFace."""
    model_id: str = Field(..., alias="modelId", description="HuggingFace model ID")
    filename: Optional[str] = Field(default=None, description="Specific filename")
    revision: str = Field(default="main", description="Git revision")

    model_config = ConfigDict(populate_by_name=True)


class DirectDownloadRequest(BaseModel):
    """Request to download from direct URL."""
    url: str = Field(..., description="Download URL")
    filename: Optional[str] = Field(default=None, description="Override filename")
    is_lora: bool = Field(default=False, alias="isLora", description="Is this a LoRA?")

    model_config = ConfigDict(populate_by_name=True)


class DownloadTaskInfo(BaseModel):
    """Information about a download task."""
    id: str = Field(..., description="Task ID")
    source: str = Field(..., description="Download source")
    name: str = Field(..., description="Model name")
    status: str = Field(..., description="Download status")
    progress: float = Field(default=0.0, description="Progress percentage")
    total_size: int = Field(default=0, alias="totalSize", description="Total size in bytes")
    downloaded_size: int = Field(default=0, alias="downloadedSize", description="Downloaded bytes")
    speed: float = Field(default=0.0, description="Speed in bytes/s")
    error: Optional[str] = Field(default=None, description="Error message if failed")

    model_config = ConfigDict(populate_by_name=True)


class DownloadListResponse(BaseModel):
    """Response listing download tasks."""
    object: str = Field(default="list")
    data: List[DownloadTaskInfo]


# =============================================================================
# AUTHENTICATION MODELS
# =============================================================================

class CreateAPIKeyRequest(BaseModel):
    """Request to create an API key."""
    name: str = Field(..., description="Friendly name for the key")
    is_admin: bool = Field(default=False, alias="isAdmin", description="Admin privileges")
    access_level: str = Field(default="user", alias="accessLevel", description="Access level: anonymous, user, admin")
    rate_limit: int = Field(default=100, alias="rateLimit", description="Requests per minute")

    model_config = ConfigDict(populate_by_name=True)


class CreateInviteRequest(BaseModel):
    """Request to create an invite code."""
    uses: int = Field(default=1, description="Number of uses (0=unlimited)")
    expires_hours: Optional[int] = Field(default=None, alias="expiresHours", description="Hours until expiration (null=never)")
    
    model_config = ConfigDict(populate_by_name=True)


class APIKeyInfo(BaseModel):
    """API key information (without sensitive data)."""
    id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    created_at: float = Field(..., alias="createdAt", description="Creation timestamp")
    last_used: Optional[float] = Field(default=None, alias="lastUsed", description="Last used timestamp")
    is_admin: bool = Field(default=False, alias="isAdmin", description="Admin flag")
    access_level: str = Field(default="user", alias="accessLevel", description="Access level")
    rate_limit: int = Field(default=100, alias="rateLimit", description="Rate limit")
    enabled: bool = Field(default=True, description="Enabled flag")

    model_config = ConfigDict(populate_by_name=True)


class APIKeyCreatedResponse(BaseModel):
    """Response after creating an API key."""
    key: str = Field(..., description="The API key (only shown once)")
    info: APIKeyInfo = Field(..., description="Key information")


class APIKeyListResponse(BaseModel):
    """Response listing API keys."""
    object: str = Field(default="list")
    data: List[APIKeyInfo]


class SessionInfo(BaseModel):
    """Session information."""
    id: str = Field(..., description="Session ID")
    api_key_id: Optional[str] = Field(default=None, alias="apiKeyId", description="Associated API key")
    created_at: float = Field(..., alias="createdAt", description="Creation timestamp")
    last_accessed: float = Field(..., alias="lastAccessed", description="Last access timestamp")
    expires_at: float = Field(..., alias="expiresAt", description="Expiration timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Session metadata")

    model_config = ConfigDict(populate_by_name=True)


class SessionCreatedResponse(BaseModel):
    """Response after creating a session."""
    session_id: str = Field(..., alias="sessionId", description="Session ID")
    token: str = Field(..., description="Session token")
    expires_at: float = Field(..., alias="expiresAt", description="Expiration timestamp")

    model_config = ConfigDict(populate_by_name=True)


class SessionListResponse(BaseModel):
    """Response listing sessions."""
    object: str = Field(default="list")
    data: List[SessionInfo]


class AuthStatsResponse(BaseModel):
    """Authentication statistics response."""
    total_keys: int = Field(default=0, alias="totalKeys")
    active_keys: int = Field(default=0, alias="activeKeys")
    admin_keys: int = Field(default=0, alias="adminKeys")
    total_sessions: int = Field(default=0, alias="totalSessions")
    active_sessions: int = Field(default=0, alias="activeSessions")

    model_config = ConfigDict(populate_by_name=True)


# =============================================================================
# GALLERY MODELS
# =============================================================================

class GalleryImageInfo(BaseModel):
    """Image information for gallery display."""
    id: str = Field(..., description="Image ID")
    filename: str = Field(..., description="Image filename")
    url: str = Field(..., description="Image URL")
    thumbnail_url: Optional[str] = Field(default=None, alias="thumbnailUrl", description="Thumbnail URL")
    is_public: bool = Field(..., alias="isPublic", description="Public visibility")
    created_at: float = Field(..., alias="createdAt", description="Creation timestamp")
    expires_at: Optional[float] = Field(default=None, alias="expiresAt", description="Expiration timestamp")
    is_owner: bool = Field(..., alias="isOwner", description="Whether requester owns this image")
    
    # Generation metadata
    prompt: str = Field(default="")
    negative_prompt: str = Field(default="", alias="negativePrompt")
    model: str = Field(default="")
    steps: int = Field(default=0)
    guidance_scale: float = Field(default=0.0, alias="guidanceScale")
    width: int = Field(default=0)
    height: int = Field(default=0)
    seed: Optional[int] = Field(default=None)
    scheduler: str = Field(default="")
    generation_time: Optional[float] = Field(default=None, alias="generationTime")
    loras: Optional[List[str]] = Field(default=None)
    lora_scales: Optional[List[float]] = Field(default=None, alias="loraScales")
    tags: Optional[List[str]] = Field(default=None, description="User-defined tags")

    model_config = ConfigDict(populate_by_name=True)


class GalleryListResponse(BaseModel):
    """Response listing gallery images."""
    object: str = Field(default="list")
    data: List[GalleryImageInfo]
    total: int = Field(..., description="Total matching images")
    limit: int = Field(..., description="Limit used")
    offset: int = Field(..., description="Offset used")


class UpdateImagePrivacyRequest(BaseModel):
    """Request to update image privacy settings."""
    is_public: bool = Field(..., alias="isPublic", description="Public visibility")
    expires_in_hours: Optional[int] = Field(default=None, alias="expiresInHours", ge=1, le=168, description="Hours until expiration (1-168, only for public images)")

    model_config = ConfigDict(populate_by_name=True)


class UpdateImageTagsRequest(BaseModel):
    """Request to update image tags."""
    tags: List[str] = Field(..., description="List of tags (replaces existing tags)")

    model_config = ConfigDict(populate_by_name=True)


class GalleryStatsResponse(BaseModel):
    """Gallery statistics response."""
    total: int = Field(..., description="Total images")
    public: int = Field(..., description="Public images")
    private: int = Field(..., alias="private", description="Private images")
    expired: int = Field(..., description="Expired images")

    model_config = ConfigDict(populate_by_name=True)

