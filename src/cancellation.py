# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Cancellation Management

Provides thread-safe cancellation tracking for long-running generation tasks.
Allows graceful cancellation of image generation without hanging the service.
"""

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class CancellationToken:
    """
    Token for tracking and signaling cancellation of a generation request.
    
    Thread-safe: Can be checked from both async and sync contexts.
    """
    request_id: str
    created_at: float = field(default_factory=time.time)
    _cancelled: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    
    def cancel(self) -> None:
        """Mark this request as cancelled."""
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                logger.info("Cancellation requested for: %s", self.request_id)
    
    def is_cancelled(self) -> bool:
        """Check if cancellation was requested (thread-safe)."""
        with self._lock:
            return self._cancelled
    
    def check_cancelled(self) -> None:
        """
        Raise CancellationError if cancelled.
        
        Use this in generation loops to abort early.
        """
        if self.is_cancelled():
            raise CancellationError(f"Request {self.request_id} was cancelled")


class CancellationError(Exception):
    """Raised when a cancellation token detects cancellation."""
    pass


class CancellationRegistry:
    """
    Registry for tracking active generation requests and their cancellation tokens.
    
    Thread-safe: Manages concurrent access from FastAPI async handlers and
    generation threads.
    """
    
    def __init__(self):
        self._tokens: Dict[str, CancellationToken] = {}
        self._lock = threading.Lock()
        logger.info("CancellationRegistry initialized")
    
    def create_token(self, request_id: Optional[str] = None) -> CancellationToken:
        """
        Create and register a new cancellation token.
        
        Args:
            request_id: Optional request ID (auto-generated if not provided)
            
        Returns:
            CancellationToken for this request
        """
        if request_id is None:
            request_id = f"gen-{uuid.uuid4().hex[:12]}"
        
        token = CancellationToken(request_id=request_id)
        
        with self._lock:
            self._tokens[request_id] = token
        
        logger.debug("Created cancellation token: %s", request_id)
        return token
    
    def get_token(self, request_id: str) -> Optional[CancellationToken]:
        """Get cancellation token by request ID."""
        with self._lock:
            return self._tokens.get(request_id)
    
    def cancel(self, request_id: str) -> bool:
        """
        Cancel a request by ID.
        
        Args:
            request_id: Request to cancel
            
        Returns:
            True if request was found and cancelled, False otherwise
        """
        with self._lock:
            token = self._tokens.get(request_id)
            if token:
                token.cancel()
                return True
            return False
    
    def unregister(self, request_id: str) -> None:
        """Remove a completed/cancelled request from registry."""
        with self._lock:
            if request_id in self._tokens:
                del self._tokens[request_id]
                logger.debug("Unregistered cancellation token: %s", request_id)
    
    def get_active_requests(self) -> Set[str]:
        """Get set of all active request IDs."""
        with self._lock:
            return set(self._tokens.keys())
    
    def get_stats(self) -> Dict[str, int]:
        """Get registry statistics."""
        with self._lock:
            return {
                "active_requests": len(self._tokens),
            }


# Global registry instance
_registry: Optional[CancellationRegistry] = None


def get_cancellation_registry() -> CancellationRegistry:
    """Get the global cancellation registry (singleton)."""
    global _registry
    if _registry is None:
        _registry = CancellationRegistry()
    return _registry
