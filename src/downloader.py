# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Model Downloader

Download models from CivitAI and HuggingFace.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import aiohttp
from huggingface_hub import snapshot_download, hf_hub_download

logger = logging.getLogger(__name__)


class DownloadStatus(str, Enum):
    """Status of a download task."""
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DownloadSource(str, Enum):
    """Source of the model download."""
    CIVITAI = "civitai"
    HUGGINGFACE = "huggingface"
    DIRECT = "direct"


@dataclass
class DownloadTask:
    """Represents a download task."""
    id: str
    source: DownloadSource
    name: str
    url: str
    destination: Path
    status: DownloadStatus = DownloadStatus.QUEUED
    progress: float = 0.0
    total_size: int = 0
    downloaded_size: int = 0
    speed: float = 0.0  # bytes per second
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "source": self.source.value,
            "name": self.name,
            "url": self.url,
            "destination": str(self.destination),
            "status": self.status.value,
            "progress": self.progress,
            "total_size": self.total_size,
            "downloaded_size": self.downloaded_size,
            "speed": self.speed,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


@dataclass
class CivitAIModel:
    """CivitAI model information."""
    id: int
    name: str
    description: str
    type: str  # Checkpoint, LoRA, etc.
    nsfw: bool
    tags: List[str]
    creator: str
    download_count: int
    rating: float
    images: List[str]
    versions: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "nsfw": self.nsfw,
            "tags": self.tags,
            "creator": self.creator,
            "download_count": self.download_count,
            "rating": self.rating,
            "images": self.images,
            "versions": self.versions,
        }


@dataclass
class HuggingFaceModel:
    """HuggingFace model information."""
    id: str
    author: str
    last_modified: str
    downloads: int
    likes: int
    tags: List[str]
    pipeline_tag: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "author": self.author,
            "last_modified": self.last_modified,
            "downloads": self.downloads,
            "likes": self.likes,
            "tags": self.tags,
            "pipeline_tag": self.pipeline_tag,
        }


class DownloadManager:
    """
    Manager for downloading models from various sources.
    
    Supports:
    - CivitAI model downloads
    - HuggingFace model downloads  
    - Direct URL downloads
    """
    
    CIVITAI_API_BASE = "https://civitai.com/api/v1"
    HUGGINGFACE_API_BASE = "https://huggingface.co/api"
    
    def __init__(
        self,
        models_dir: Path,
        loras_dir: Optional[Path] = None,
        max_concurrent: int = 2,
        civitai_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
    ):
        """
        Initialize the download manager.
        
        Args:
            models_dir: Directory for downloading models
            loras_dir: Directory for downloading LoRAs (defaults to models_dir/loras)
            max_concurrent: Maximum concurrent downloads
            civitai_api_key: Optional CivitAI API key for authentication
            huggingface_token: Optional HuggingFace token for private models
        """
        self.models_dir = Path(models_dir)
        self.loras_dir = loras_dir or self.models_dir / "loras"
        self.max_concurrent = max_concurrent
        self.civitai_api_key = civitai_api_key or os.environ.get("CIVITAI_API_KEY")
        self.huggingface_token = huggingface_token or os.environ.get("HF_TOKEN")
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loras_dir.mkdir(parents=True, exist_ok=True)
        
        # Download queue and active downloads
        self._tasks: Dict[str, DownloadTask] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._active_downloads = 0
        self._workers: List[asyncio.Task] = []
        self._running = False
        
        # Progress callbacks
        self._progress_callbacks: List[Callable[[DownloadTask], None]] = []
        
        # Completion callbacks (called when a download finishes successfully)
        self._completion_callbacks: List[Callable[[], None]] = []
        
        logger.info(
            "DownloadManager initialized: models_dir=%s, loras_dir=%s",
            self.models_dir, self.loras_dir
        )
    
    async def start(self) -> None:
        """Start the download manager workers."""
        if self._running:
            return
        
        self._running = True
        
        # Start worker tasks
        for i in range(self.max_concurrent):
            worker = asyncio.create_task(self._download_worker(i))
            self._workers.append(worker)
        
        logger.info("Download manager started with %d workers", self.max_concurrent)
    
    async def stop(self) -> None:
        """Stop the download manager."""
        self._running = False
        
        # Cancel all workers
        for worker in self._workers:
            worker.cancel()
        
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        self._workers.clear()
        logger.info("Download manager stopped")
    
    def on_progress(self, callback: Callable[[DownloadTask], None]) -> None:
        """Register a progress callback."""
        self._progress_callbacks.append(callback)
    
    def on_complete(self, callback: Callable[[], None]) -> None:
        """Register a completion callback (called when any download finishes successfully)."""
        self._completion_callbacks.append(callback)
    
    def _notify_progress(self, task: DownloadTask) -> None:
        """Notify all progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(task)
            except Exception as e:
                logger.warning("Progress callback error: %s", e)
    
    # =========================================================================
    # CIVITAI API
    # =========================================================================
    
    async def search_civitai(
        self,
        query: str,
        types: Optional[List[str]] = None,
        sort: str = "Highest Rated",
        period: str = "AllTime",
        nsfw: bool = True,
        limit: int = 100,
        page: int = 1,
    ) -> List[CivitAIModel]:
        """
        Search CivitAI for models.
        
        CivitAI API behavior:
        - With query: search algorithm is used, types filter doesn't work well
        - Without query: browse mode, types filter works, pagination available
        
        Args:
            query: Search query (empty string = browse mode)
            types: Model types filter (Checkpoint, LORA, etc.)
            sort: Sort order (Highest Rated, Most Downloaded, Newest)
            period: Time period for sorting
            nsfw: Include NSFW models
            limit: Maximum results
            page: Page number (only works without query)
            
        Returns:
            List of CivitAI models
        """
        params = {
            "sort": sort,
            "period": period,
            "nsfw": str(nsfw).lower(),
            "limit": limit,
        }
        
        # CivitAI API quirk: query and page params don't work together
        # Also, types filter only works reliably in browse mode (no query)
        if query and query.strip():
            params["query"] = query.strip()
            # Client-side filtering when using query (API types filter is buggy)
            use_client_side_filter = True
        else:
            # Browse mode - server-side type filter works, pagination available
            params["page"] = page
            if types:
                params["types"] = ",".join(types)
            use_client_side_filter = False
        
        headers = {}
        if self.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.civitai_api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.CIVITAI_API_BASE}/models",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error("CivitAI search failed: %s", response.status)
                        return []
                    
                    data = await response.json()
                    models = []
                    
                    for item in data.get("items", []):
                        try:
                            model_type = item.get("type", "Unknown")
                            
                            # Client-side type filtering only when searching (not browsing)
                            if use_client_side_filter and types and model_type not in types:
                                continue
                            
                            model = CivitAIModel(
                                id=item["id"],
                                name=item["name"],
                                description=item.get("description", "")[:500],
                                type=model_type,
                                nsfw=item.get("nsfw", False),
                                tags=item.get("tags", []),
                                creator=item.get("creator", {}).get("username", "Unknown"),
                                download_count=item.get("stats", {}).get("downloadCount", 0),
                                rating=item.get("stats", {}).get("rating", 0),
                                images=[
                                    img.get("url", "") 
                                    for img in item.get("modelVersions", [{}])[0].get("images", [])[:3]
                                ],
                                versions=[
                                    {
                                        "id": v["id"],
                                        "name": v["name"],
                                        "files": [
                                            {
                                                "id": f["id"],
                                                "name": f["name"],
                                                "size": f.get("sizeKB", 0) * 1024,
                                                "type": f.get("type", "Model"),
                                                "downloadUrl": f.get("downloadUrl", ""),
                                            }
                                            for f in v.get("files", [])
                                        ],
                                    }
                                    for v in item.get("modelVersions", [])[:3]
                                ],
                            )
                            models.append(model)
                        except (KeyError, IndexError) as e:
                            logger.debug("Skipping malformed model: %s", e)
                            continue
                    
                    logger.info("CivitAI search returned %d models", len(models))
                    return models
                    
        except Exception as e:
            logger.error("CivitAI search error: %s", e)
            return []
    
    async def download_civitai(
        self,
        model_id: int,
        version_id: Optional[int] = None,
        file_id: Optional[int] = None,
        model_type: str = "Checkpoint",
    ) -> Optional[str]:
        """
        Queue download of a CivitAI model.
        
        Args:
            model_id: CivitAI model ID
            version_id: Specific version ID (uses latest if not specified)
            file_id: Specific file ID (uses primary if not specified)
            model_type: Type of model for determining destination
            
        Returns:
            Download task ID or None if failed
        """
        headers = {}
        if self.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.civitai_api_key}"
        
        try:
            # Get model info
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.CIVITAI_API_BASE}/models/{model_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error("Failed to get CivitAI model info: %s", response.status)
                        return None
                    
                    data = await response.json()
            
            # Get version
            versions = data.get("modelVersions", [])
            if not versions:
                logger.error("No versions found for model %d", model_id)
                return None
            
            if version_id:
                version = next((v for v in versions if v["id"] == version_id), None)
                if not version:
                    logger.error("Version %d not found for model %d", version_id, model_id)
                    return None
            else:
                version = versions[0]  # Latest version
            
            # Get file
            files = version.get("files", [])
            if not files:
                logger.error("No files found for version %d", version["id"])
                return None
            
            if file_id:
                file_info = next((f for f in files if f["id"] == file_id), None)
                if not file_info:
                    logger.error("File %d not found", file_id)
                    return None
            else:
                # Find primary file (usually the .safetensors model)
                file_info = next(
                    (f for f in files if f.get("type") == "Model"),
                    files[0]
                )
            
            # Build download URL with API key if available
            download_url = file_info.get("downloadUrl", "")
            if self.civitai_api_key:
                separator = "&" if "?" in download_url else "?"
                download_url = f"{download_url}{separator}token={self.civitai_api_key}"
            
            # Determine destination directory
            actual_type = data.get("type", model_type)
            if actual_type.upper() == "LORA":
                dest_dir = self.loras_dir
            else:
                dest_dir = self.models_dir
            
            # Clean filename
            filename = file_info.get("name", f"{data['name']}.safetensors")
            filename = self._sanitize_filename(filename)
            
            # Create download task
            task_id = str(uuid.uuid4())[:8]
            task = DownloadTask(
                id=task_id,
                source=DownloadSource.CIVITAI,
                name=data.get("name", f"Model {model_id}"),
                url=download_url,
                destination=dest_dir / filename,
                total_size=int(file_info.get("sizeKB", 0) * 1024),
                metadata={
                    "civitai_id": model_id,
                    "version_id": version["id"],
                    "file_id": file_info["id"],
                    "type": actual_type,
                },
            )
            
            self._tasks[task_id] = task
            await self._queue.put(task_id)
            
            logger.info("Queued CivitAI download: %s -> %s", task.name, task.destination)
            return task_id
            
        except Exception as e:
            logger.error("Failed to queue CivitAI download: %s", e)
            return None
    
    # =========================================================================
    # HUGGINGFACE API
    # =========================================================================
    
    async def search_huggingface(
        self,
        query: str,
        filter_tags: Optional[List[str]] = None,
        sort: str = "downloads",
        limit: int = 100,
    ) -> List[HuggingFaceModel]:
        """
        Search HuggingFace for models.
        
        Args:
            query: Search query
            filter_tags: Tags to filter by (e.g., ["diffusers", "stable-diffusion"])
            sort: Sort field (downloads, likes, created)
            limit: Maximum results
            
        Returns:
            List of HuggingFace models
        """
        params = {
            "search": query,
            "sort": sort,
            "direction": "-1",
            "limit": limit,
        }
        
        # Add diffusion-related filters by default
        if filter_tags:
            params["filter"] = ",".join(filter_tags)
        else:
            params["filter"] = "diffusers"
        
        headers = {}
        if self.huggingface_token:
            headers["Authorization"] = f"Bearer {self.huggingface_token}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.HUGGINGFACE_API_BASE}/models",
                    params=params,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error("HuggingFace search failed: %s", response.status)
                        return []
                    
                    data = await response.json()
                    models = []
                    
                    for item in data:
                        try:
                            model = HuggingFaceModel(
                                id=item["id"],
                                author=item.get("author", item["id"].split("/")[0] if "/" in item["id"] else "unknown"),
                                last_modified=item.get("lastModified", ""),
                                downloads=item.get("downloads", 0),
                                likes=item.get("likes", 0),
                                tags=item.get("tags", []),
                                pipeline_tag=item.get("pipeline_tag"),
                            )
                            models.append(model)
                        except (KeyError, IndexError) as e:
                            logger.debug("Skipping malformed model: %s", e)
                            continue
                    
                    logger.info("HuggingFace search returned %d models", len(models))
                    return models
                    
        except Exception as e:
            logger.error("HuggingFace search error: %s", e)
            return []
    
    async def download_huggingface(
        self,
        model_id: str,
        filename: Optional[str] = None,
        revision: str = "main",
    ) -> Optional[str]:
        """
        Queue download of a HuggingFace model.
        
        For diffuser models (directories), this clones the entire repo.
        For single-file models, downloads just that file.
        
        Args:
            model_id: HuggingFace model ID (e.g., "runwayml/stable-diffusion-v1-5")
            filename: Specific file to download (clones repo if not specified)
            revision: Git revision/branch
            
        Returns:
            Download task ID or None if failed
        """
        headers = {}
        if self.huggingface_token:
            headers["Authorization"] = f"Bearer {self.huggingface_token}"
        
        try:
            # Get model info to determine if it's a diffusers model
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.HUGGINGFACE_API_BASE}/models/{model_id}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.error("Failed to get HuggingFace model info: %s", response.status)
                        return None
                    
                    model_info = await response.json()
            
            # Check if this is a diffusers model (has model_index.json or pipeline_tag indicates it)
            is_diffusers_model = False
            tags = model_info.get("tags", [])
            pipeline_tag = model_info.get("pipeline_tag", "")
            siblings = model_info.get("siblings", [])
            
            # Check for diffusers indicators
            if "diffusers" in tags or pipeline_tag in ["text-to-image", "image-to-image"]:
                is_diffusers_model = True
            
            # Also check if model_index.json exists (definitive diffusers indicator)
            for sibling in siblings:
                if sibling.get("rfilename") == "model_index.json":
                    is_diffusers_model = True
                    break
            
            model_name = model_id.split("/")[-1]
            
            if is_diffusers_model and not filename:
                # For diffusers models, clone the entire repository
                logger.info("Detected diffusers model, will clone entire repo: %s", model_id)
                
                # Build clone URL
                clone_url = f"https://huggingface.co/{model_id}"
                
                # Create download task for repo clone
                task_id = str(uuid.uuid4())[:8]
                task = DownloadTask(
                    id=task_id,
                    source=DownloadSource.HUGGINGFACE,
                    name=model_name,
                    url=clone_url,
                    destination=self.models_dir / model_name,
                    total_size=0,  # Unknown for git clone
                    metadata={
                        "huggingface_id": model_id,
                        "revision": revision,
                        "is_diffusers": True,
                        "clone_repo": True,
                    },
                )
                
                self._tasks[task_id] = task
                await self._queue.put(task_id)
                
                logger.info("Queued HuggingFace repo clone: %s -> %s", task.name, task.destination)
                return task_id
            
            else:
                # For single file downloads, get the file list
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.HUGGINGFACE_API_BASE}/models/{model_id}/tree/{revision}",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        if response.status != 200:
                            logger.error("Failed to get HuggingFace model tree: %s", response.status)
                            return None
                        
                        files = await response.json()
                
                # Find the file to download
                if filename:
                    file_info = next((f for f in files if f.get("path") == filename), None)
                else:
                    # Look for safetensors files first, then ckpt
                    file_info = next(
                        (f for f in files if f.get("path", "").endswith(".safetensors") and f.get("type") == "file"),
                        next(
                            (f for f in files if f.get("path", "").endswith(".ckpt") and f.get("type") == "file"),
                            None
                        )
                    )
                
                if not file_info:
                    logger.error("No suitable model file found in %s", model_id)
                    return None
                
                # Build download URL
                file_path = file_info["path"]
                download_url = f"https://huggingface.co/{model_id}/resolve/{revision}/{file_path}"
                
                # Determine destination
                dest_filename = self._sanitize_filename(file_path.split("/")[-1])
                
                # Create download task
                task_id = str(uuid.uuid4())[:8]
                task = DownloadTask(
                    id=task_id,
                    source=DownloadSource.HUGGINGFACE,
                    name=model_name,
                    url=download_url,
                    destination=self.models_dir / dest_filename,
                    total_size=file_info.get("size", 0),
                    metadata={
                        "huggingface_id": model_id,
                        "revision": revision,
                        "file_path": file_path,
                        "is_diffusers": False,
                    },
                )
                
                self._tasks[task_id] = task
                await self._queue.put(task_id)
                
                logger.info("Queued HuggingFace download: %s -> %s", task.name, task.destination)
                return task_id
            
        except Exception as e:
            logger.error("Failed to queue HuggingFace download: %s", e)
            return None
    
    # =========================================================================
    # DIRECT DOWNLOAD
    # =========================================================================
    
    async def download_url(
        self,
        url: str,
        filename: Optional[str] = None,
        is_lora: bool = False,
    ) -> Optional[str]:
        """
        Queue download from a direct URL.
        
        Args:
            url: Direct download URL
            filename: Override filename (extracted from URL if not specified)
            is_lora: Whether this is a LoRA (determines destination)
            
        Returns:
            Download task ID or None if failed
        """
        # Extract filename from URL if not provided
        if not filename:
            parsed = urlparse(url)
            filename = os.path.basename(parsed.path)
            if not filename:
                filename = f"model_{uuid.uuid4().hex[:8]}.safetensors"
        
        filename = self._sanitize_filename(filename)
        dest_dir = self.loras_dir if is_lora else self.models_dir
        
        task_id = str(uuid.uuid4())[:8]
        task = DownloadTask(
            id=task_id,
            source=DownloadSource.DIRECT,
            name=filename,
            url=url,
            destination=dest_dir / filename,
            metadata={"is_lora": is_lora},
        )
        
        self._tasks[task_id] = task
        await self._queue.put(task_id)
        
        logger.info("Queued direct download: %s -> %s", url, task.destination)
        return task_id
    
    # =========================================================================
    # TASK MANAGEMENT
    # =========================================================================
    
    def get_task(self, task_id: str) -> Optional[DownloadTask]:
        """Get a download task by ID."""
        return self._tasks.get(task_id)
    
    def list_tasks(self) -> List[DownloadTask]:
        """List all download tasks."""
        return list(self._tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a download task.
        
        Args:
            task_id: Task ID to cancel
            
        Returns:
            True if cancelled, False if not found or already complete
        """
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        if task.status in (DownloadStatus.COMPLETED, DownloadStatus.FAILED):
            return False
        
        task.status = DownloadStatus.CANCELLED
        self._notify_progress(task)
        logger.info("Cancelled download: %s", task_id)
        return True
    
    def clear_completed(self) -> int:
        """
        Clear completed and failed tasks.
        
        Returns:
            Number of tasks cleared
        """
        to_remove = [
            task_id
            for task_id, task in self._tasks.items()
            if task.status in (DownloadStatus.COMPLETED, DownloadStatus.FAILED, DownloadStatus.CANCELLED)
        ]
        
        for task_id in to_remove:
            del self._tasks[task_id]
        
        return len(to_remove)
    
    # =========================================================================
    # INTERNAL METHODS
    # =========================================================================
    
    async def _download_worker(self, worker_id: int) -> None:
        """Background worker for processing downloads."""
        logger.debug("Download worker %d started", worker_id)
        
        while self._running:
            try:
                # Get task from queue with timeout
                try:
                    task_id = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                task = self._tasks.get(task_id)
                if not task or task.status == DownloadStatus.CANCELLED:
                    self._queue.task_done()
                    continue
                
                # Process download
                await self._process_download(task)
                self._queue.task_done()
                
                # Notify completion callbacks if download succeeded
                if task.status == DownloadStatus.COMPLETED:
                    for callback in self._completion_callbacks:
                        try:
                            callback()
                        except Exception as e:
                            logger.warning("Completion callback error: %s", e)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Download worker %d error: %s", worker_id, e)
        
        logger.debug("Download worker %d stopped", worker_id)
    
    async def _process_download(self, task: DownloadTask) -> None:
        """Process a single download task."""
        task.status = DownloadStatus.DOWNLOADING
        task.started_at = time.time()
        self._notify_progress(task)
        
        # Check if this is a git clone task for diffusers models
        if task.metadata.get("clone_repo"):
            await self._process_git_clone(task)
            return
        
        headers = {}
        if task.source == DownloadSource.HUGGINGFACE and self.huggingface_token:
            headers["Authorization"] = f"Bearer {self.huggingface_token}"
        
        try:
            # Ensure destination directory exists
            task.destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress tracking
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    task.url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=3600),  # 1 hour timeout
                ) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}: {response.reason}")
                    
                    # Get total size from headers if not known
                    if task.total_size == 0:
                        content_length = response.headers.get("content-length")
                        if content_length:
                            task.total_size = int(content_length)
                    
                    # Download file
                    temp_path = task.destination.with_suffix(".tmp")
                    chunk_size = 1024 * 1024  # 1MB chunks
                    last_update = time.time()
                    bytes_since_update = 0
                    
                    with open(temp_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(chunk_size):
                            if task.status == DownloadStatus.CANCELLED:
                                raise Exception("Download cancelled")
                            
                            f.write(chunk)
                            task.downloaded_size += len(chunk)
                            bytes_since_update += len(chunk)
                            
                            # Update progress
                            if task.total_size > 0:
                                task.progress = (task.downloaded_size / task.total_size) * 100
                            
                            # Calculate speed every second
                            now = time.time()
                            elapsed = now - last_update
                            if elapsed >= 1.0:
                                task.speed = bytes_since_update / elapsed
                                last_update = now
                                bytes_since_update = 0
                                self._notify_progress(task)
                    
                    # Move temp file to final destination
                    temp_path.rename(task.destination)
            
            # Success
            task.status = DownloadStatus.COMPLETED
            task.progress = 100.0
            task.completed_at = time.time()
            
            logger.info(
                "Download completed: %s (%.1f MB in %.1fs)",
                task.name,
                task.downloaded_size / (1024 * 1024),
                task.completed_at - task.started_at
            )
            
        except Exception as e:
            task.status = DownloadStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            logger.error("Download failed: %s - %s", task.name, e)
            
            # Clean up temp file
            temp_path = task.destination.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()
        
        self._notify_progress(task)
    
    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of a directory in bytes."""
        total = 0
        try:
            for root, dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    async def _monitor_download_progress(self, task: DownloadTask, destination: Path) -> None:
        """Monitor directory size growth and update task progress during snapshot_download.
        
        Runs concurrently with the download. Estimates progress by checking the
        expected total size from the HuggingFace API, or by tracking size growth rate.
        Progress is reported between 5% and 95% (5% = started, 95% = nearly done,
        100% is set only when the download is confirmed complete).
        """
        last_size = 0
        stall_count = 0
        
        # Try to get expected total size from HuggingFace API
        expected_size = task.metadata.get("expected_size", 0)
        
        while task.status == DownloadStatus.DOWNLOADING:
            await asyncio.sleep(3)  # Check every 3 seconds
            
            if task.status != DownloadStatus.DOWNLOADING:
                break
            
            current_size = self._get_dir_size(destination)
            task.downloaded_size = current_size
            
            # Calculate speed
            size_delta = current_size - last_size
            if size_delta > 0:
                task.speed = size_delta / 3.0  # bytes per second (3s interval)
                stall_count = 0
            else:
                stall_count += 1
            
            # Update progress
            if expected_size > 0:
                # We know the expected size - calculate real percentage
                raw_pct = (current_size / expected_size) * 100.0
                # Clamp between 5% and 95% (100% only set on confirmed completion)
                task.progress = min(95.0, max(5.0, raw_pct))
                task.total_size = expected_size
            else:
                # Unknown total size - use a logarithmic curve that approaches 95%
                # This gives meaningful progress even without knowing the total
                if current_size > 0:
                    # Each GB downloaded adds less progress (diminishing returns)
                    gb_downloaded = current_size / (1024 ** 3)
                    # Asymptotic curve: approaches 95% as size grows
                    task.progress = min(95.0, 5.0 + 90.0 * (1.0 - 1.0 / (1.0 + gb_downloaded / 5.0)))
            
            self._notify_progress(task)
            last_size = current_size
            
            if current_size > 0:
                logger.debug(
                    "Download progress: %s - %.1f MB (%.1f%%, %.1f MB/s)",
                    task.name,
                    current_size / (1024 * 1024),
                    task.progress,
                    (task.speed or 0) / (1024 * 1024),
                )

    async def _process_git_clone(self, task: DownloadTask) -> None:
        """Process a HuggingFace diffusers model download using huggingface_hub."""
        try:
            model_id = task.metadata.get("huggingface_id", "")
            revision = task.metadata.get("revision", "main")
            destination = task.destination
            
            # Remove existing destination if it exists
            if destination.exists():
                logger.info("Removing existing directory: %s", destination)
                shutil.rmtree(destination)
            
            logger.info("Downloading HuggingFace model: %s -> %s", model_id, destination)
            
            # Try to get expected total size from HuggingFace API for accurate progress
            try:
                headers = {}
                if self.huggingface_token:
                    headers["Authorization"] = f"Bearer {self.huggingface_token}"
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.HUGGINGFACE_API_BASE}/models/{model_id}",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=15),
                    ) as response:
                        if response.status == 200:
                            model_info = await response.json()
                            siblings = model_info.get("siblings", [])
                            expected_size = sum(s.get("size", 0) for s in siblings if s.get("size"))
                            if expected_size > 0:
                                task.metadata["expected_size"] = expected_size
                                task.total_size = expected_size
                                logger.info("Expected download size for %s: %.1f MB", model_id, expected_size / (1024 * 1024))
            except Exception as e:
                logger.debug("Could not get expected size from HuggingFace API: %s", e)
            
            task.progress = 5.0
            self._notify_progress(task)
            
            # Use huggingface_hub's snapshot_download in a thread pool
            # to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            
            def do_download():
                return snapshot_download(
                    repo_id=model_id,
                    revision=revision,
                    local_dir=str(destination),
                    local_dir_use_symlinks=False,
                    token=self.huggingface_token,
                    resume_download=True,
                )
            
            # Start progress monitor and download concurrently
            # The monitor tracks directory size growth while snapshot_download runs
            monitor_task = asyncio.create_task(self._monitor_download_progress(task, destination))
            
            try:
                await loop.run_in_executor(None, do_download)
            finally:
                # Stop the monitor
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            
            # Calculate final total size
            total_size = self._get_dir_size(destination)
            
            task.downloaded_size = total_size
            task.total_size = total_size
            task.progress = 100.0
            task.speed = 0
            task.status = DownloadStatus.COMPLETED
            task.completed_at = time.time()
            
            logger.info(
                "HuggingFace model download completed: %s (%.1f MB in %.1fs)",
                task.name,
                total_size / (1024 * 1024),
                task.completed_at - task.started_at
            )
            
        except Exception as e:
            task.status = DownloadStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            logger.error("HuggingFace download failed: %s - %s", task.name, e)
            
            # Clean up partial download
            if task.destination.exists():
                try:
                    shutil.rmtree(task.destination)
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up partial download: %s", cleanup_error)
        
        self._notify_progress(task)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe file system use."""
        # Remove or replace unsafe characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove leading/trailing spaces and dots
        filename = filename.strip('. ')
        # Ensure it's not empty
        if not filename:
            filename = "model.safetensors"
        return filename
