# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Image Gallery System

Manages image metadata, ownership, privacy settings, and expiration.
Uses JSON file storage (consistent with auth system).
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    """Image metadata record with privacy and ownership."""
    id: str  # UUID matching image filename (without extension)
    filename: str  # Full filename (e.g., "abc123.png")
    owner_api_key_id: Optional[str]  # API key ID that created this image
    is_public: bool = False  # Whether image is publicly visible
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None  # Optional expiration timestamp for public images
    
    # Generation metadata
    prompt: str = ""
    negative_prompt: str = ""
    model: str = ""
    steps: int = 0
    guidance_scale: float = 0.0
    width: int = 0
    height: int = 0
    seed: Optional[int] = None
    scheduler: str = ""
    generation_time: Optional[float] = None  # Total generation time in seconds
    loras: Optional[List[str]] = None
    lora_scales: Optional[List[float]] = None
    tags: Optional[List[str]] = None  # User-defined tags for organization
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "ImageRecord":
        """Create from dictionary."""
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if image has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def is_accessible_by(self, api_key_id: Optional[str], is_admin: bool = False) -> bool:
        """
        Check if image is accessible by the given API key.
        
        Rules:
        - Owner can always access their own images
        - Public non-expired images are accessible to everyone
        - Private images are ONLY accessible to owner
        - Admins do NOT get special access to private images
        """
        # Owner can always access their own images
        if api_key_id and self.owner_api_key_id == api_key_id:
            return True
        
        # Public non-expired images are accessible
        if self.is_public and not self.is_expired():
            return True
        
        return False


class GalleryManager:
    """
    Manages the image gallery storage.
    
    Uses JSON file storage for simplicity (no database required).
    Thread-safe for concurrent access.
    """
    
    def __init__(self, storage_path: Path):
        """
        Initialize the gallery manager.
        
        Args:
            storage_path: Path to the gallery.json file
        """
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._images: Dict[str, ImageRecord] = {}
        self._load()
    
    def _load(self) -> None:
        """Load gallery from JSON file."""
        if not self.storage_path.exists():
            logger.info("Gallery file not found, starting fresh: %s", self.storage_path)
            self._save()
            return
        
        try:
            with open(self.storage_path, "r") as f:
                data = json.load(f)
            
            self._images = {
                img_id: ImageRecord.from_dict(img_data)
                for img_id, img_data in data.get("images", {}).items()
            }
            
            logger.info("Loaded %d images from gallery", len(self._images))
        except Exception as e:
            logger.error("Failed to load gallery: %s", e)
            self._images = {}
    
    def _save(self) -> None:
        """Save gallery to JSON file."""
        try:
            data = {
                "images": {
                    img_id: img.to_dict()
                    for img_id, img in self._images.items()
                }
            }
            
            # Write atomically
            temp_path = self.storage_path.with_suffix(".tmp")
            with open(temp_path, "w") as f:
                json.dump(data, f, indent=2)
            temp_path.replace(self.storage_path)
            
            logger.debug("Saved gallery with %d images", len(self._images))
        except Exception as e:
            logger.error("Failed to save gallery: %s", e)
    
    def add_image(self, image: ImageRecord) -> None:
        """Add an image to the gallery."""
        with self._lock:
            self._images[image.id] = image
            self._save()
            logger.info("Added image to gallery: %s (owner=%s, public=%s)", 
                       image.id, image.owner_api_key_id, image.is_public)
    
    def get_image(self, image_id: str) -> Optional[ImageRecord]:
        """Get an image by ID."""
        with self._lock:
            return self._images.get(image_id)
    
    def update_privacy(self, image_id: str, is_public: bool, expires_at: Optional[float] = None) -> bool:
        """
        Update image privacy settings.
        
        Args:
            image_id: Image ID
            is_public: Whether image should be public
            expires_at: Optional expiration timestamp (only for public images)
        
        Returns:
            True if updated successfully, False if image not found
        """
        with self._lock:
            image = self._images.get(image_id)
            if not image:
                return False
            
            image.is_public = is_public
            image.expires_at = expires_at if is_public else None
            self._save()
            
            logger.info("Updated image privacy: %s (public=%s, expires=%s)",
                       image_id, is_public, expires_at)
            return True
    
    def update_tags(self, image_id: str, tags: List[str]) -> bool:
        """
        Update image tags.
        
        Args:
            image_id: Image ID
            tags: List of tags (replaces existing tags)
        
        Returns:
            True if updated successfully, False if image not found
        """
        with self._lock:
            image = self._images.get(image_id)
            if not image:
                return False
            
            image.tags = tags if tags else None
            self._save()
            
            logger.info("Updated image tags: %s (tags=%s)", image_id, tags)
            return True
    
    def delete_image(self, image_id: str) -> bool:
        """
        Delete an image from the gallery.
        
        Args:
            image_id: Image ID
        
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if image_id in self._images:
                del self._images[image_id]
                self._save()
                logger.info("Deleted image from gallery: %s", image_id)
                return True
            return False
    
    def _filter_accessible(
        self,
        api_key_id: Optional[str] = None,
        is_admin: bool = False,
        include_public: bool = True,
        include_private: bool = True,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
    ) -> List[ImageRecord]:
        """Return all accessible images matching filters, sorted newest first.

        Must be called while holding self._lock.
        """
        accessible = []

        for image in self._images.values():
            if image.is_public and image.is_expired():
                continue
            if not image.is_accessible_by(api_key_id, is_admin):
                continue

            is_own_image = api_key_id and image.owner_api_key_id == api_key_id

            if image.is_public and not include_public:
                if not is_own_image:
                    continue
            if not image.is_public and not include_private:
                continue

            if tags:
                image_tags = set(image.tags or [])
                if not all(tag in image_tags for tag in tags):
                    continue

            if search:
                if search.lower() not in image.prompt.lower():
                    continue

            accessible.append(image)

        accessible.sort(key=lambda x: x.created_at, reverse=True)
        return accessible

    def list_images(
        self,
        api_key_id: Optional[str] = None,
        is_admin: bool = False,
        include_public: bool = True,
        include_private: bool = True,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ImageRecord]:
        """
        List images accessible to the given user.
        
        Args:
            api_key_id: API key ID of the requester (None for anonymous)
            is_admin: Whether requester is an admin
            include_public: Include public images
            include_private: Include private images (only owner's)
            tags: Filter by tags (images must have ALL specified tags)
            search: Search prompt text (case-insensitive)
            limit: Maximum number of images to return (0 = return all)
            offset: Offset for pagination
        
        Returns:
            List of accessible images, sorted by created_at (newest first)
        """
        with self._lock:
            accessible = self._filter_accessible(
                api_key_id=api_key_id,
                is_admin=is_admin,
                include_public=include_public,
                include_private=include_private,
                tags=tags,
                search=search,
            )
            if limit > 0:
                return accessible[offset:offset + limit]
            return accessible[offset:]

    def count_images(
        self,
        api_key_id: Optional[str] = None,
        is_admin: bool = False,
        include_public: bool = True,
        include_private: bool = True,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
    ) -> int:
        """Return total count of accessible images matching filters."""
        with self._lock:
            return len(self._filter_accessible(
                api_key_id=api_key_id,
                is_admin=is_admin,
                include_public=include_public,
                include_private=include_private,
                tags=tags,
                search=search,
            ))
    
    def cleanup_expired(self, images_dir: Path) -> int:
        """
        Clean up expired public images.
        
        Args:
            images_dir: Directory containing image files
        
        Returns:
            Number of images cleaned up
        """
        with self._lock:
            expired_ids = [
                img_id for img_id, img in self._images.items()
                if img.is_expired()
            ]
            
            cleaned = 0
            for img_id in expired_ids:
                image = self._images[img_id]
                
                # Delete image file
                image_path = images_dir / image.filename
                if image_path.exists():
                    try:
                        image_path.unlink()
                        logger.info("Deleted expired image file: %s", image.filename)
                    except Exception as e:
                        logger.error("Failed to delete expired image file %s: %s", 
                                   image.filename, e)
                        continue
                
                # Remove from gallery
                del self._images[img_id]
                cleaned += 1
            
            if cleaned > 0:
                self._save()
                logger.info("Cleaned up %d expired images", cleaned)
            
            return cleaned
    
    def get_stats(self) -> Dict:
        """Get gallery statistics."""
        with self._lock:
            total = len(self._images)
            public = sum(1 for img in self._images.values() if img.is_public)
            private = total - public
            expired = sum(1 for img in self._images.values() if img.is_expired())
            
            return {
                "total": total,
                "public": public,
                "private": private,
                "expired": expired
            }
