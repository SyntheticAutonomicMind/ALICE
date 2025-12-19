# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Authentication System

Simple API key and session management using JSON file storage.
Supports three access levels:
- admin: Full control (manage keys, sessions, models)
- user: Can generate images, view models, download
- anonymous: Can view models, basic health checks
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from threading import Lock

logger = logging.getLogger(__name__)


# Default session timeout configuration (in seconds)
# These can be overridden by config.server.session_timeout_seconds
DEFAULT_SESSION_TIMEOUT_SECONDS = 900  # 15 minutes
DEFAULT_SESSION_INACTIVITY_TIMEOUT_SECONDS = 900  # 15 minutes of inactivity

# Runtime timeout values (can be updated from config)
SESSION_TIMEOUT_SECONDS = DEFAULT_SESSION_TIMEOUT_SECONDS
SESSION_INACTIVITY_TIMEOUT_SECONDS = DEFAULT_SESSION_INACTIVITY_TIMEOUT_SECONDS


def set_session_timeout(timeout_seconds: int) -> None:
    """Set the session timeout values (called from main on startup)."""
    global SESSION_TIMEOUT_SECONDS, SESSION_INACTIVITY_TIMEOUT_SECONDS
    SESSION_TIMEOUT_SECONDS = timeout_seconds
    SESSION_INACTIVITY_TIMEOUT_SECONDS = timeout_seconds
    logger.info("Session timeout set to %d seconds", timeout_seconds)


class AccessLevel(str, Enum):
    """User access levels."""
    ANONYMOUS = "anonymous"  # Can view models, health checks
    USER = "user"           # Can generate, download, view
    ADMIN = "admin"         # Full control


def generate_api_key() -> str:
    """Generate a secure API key."""
    return f"sd-{secrets.token_urlsafe(32)}"


def hash_key(key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(key.encode()).hexdigest()


def verify_key(key: str, hashed: str) -> bool:
    """Verify an API key against its hash."""
    return hmac.compare_digest(hash_key(key), hashed)


@dataclass
class APIKey:
    """API key record."""
    id: str
    name: str
    key_hash: str
    created_at: float
    last_used: Optional[float] = None
    is_admin: bool = False
    access_level: str = "user"  # anonymous, user, admin
    rate_limit: int = 100  # requests per minute
    enabled: bool = True
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "APIKey":
        # Handle migration from old format without access_level
        if "access_level" not in data:
            data["access_level"] = "admin" if data.get("is_admin") else "user"
        return cls(**data)
    
    def get_access_level(self) -> AccessLevel:
        """Get the effective access level."""
        if self.is_admin or self.access_level == "admin":
            return AccessLevel.ADMIN
        elif self.access_level == "user":
            return AccessLevel.USER
        return AccessLevel.ANONYMOUS


@dataclass
class Session:
    """User session record."""
    id: str
    token: str
    api_key_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + SESSION_TIMEOUT_SECONDS)  # 15 minutes
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        return cls(**data)
    
    def is_expired(self) -> bool:
        """Check if session is expired (absolute timeout or inactivity)."""
        now = time.time()
        # Check absolute expiry time
        if now > self.expires_at:
            return True
        # Check inactivity timeout (15 minutes since last access)
        if now - self.last_accessed > SESSION_INACTIVITY_TIMEOUT_SECONDS:
            return True
        return False
    
    def refresh(self) -> None:
        """Refresh the session (update last_accessed and extend expiry)."""
        self.last_accessed = time.time()
        self.expires_at = self.last_accessed + SESSION_TIMEOUT_SECONDS
    
    def time_remaining(self) -> float:
        """Get seconds remaining until session expires."""
        now = time.time()
        time_since_access = now - self.last_accessed
        inactivity_remaining = SESSION_INACTIVITY_TIMEOUT_SECONDS - time_since_access
        absolute_remaining = self.expires_at - now
        return max(0, min(inactivity_remaining, absolute_remaining))


@dataclass
class InviteCode:
    """Invite code for user registration."""
    code: str
    created_by: str  # admin key id
    created_at: float
    expires_at: Optional[float] = None  # None = never expires
    uses_remaining: int = 1  # How many times it can be used
    used_by: List[str] = field(default_factory=list)  # List of key IDs created with this code
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "InviteCode":
        return cls(**data)
    
    def is_valid(self) -> bool:
        # uses_remaining of -1 means unlimited, 0 means exhausted
        if self.uses_remaining == 0:
            return False
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True


@dataclass
class PendingRegistration:
    """Pending user registration awaiting approval."""
    id: str
    name: str
    created_at: float
    status: str = "pending"  # pending, approved, rejected
    approved_by: Optional[str] = None  # admin key id
    approved_at: Optional[float] = None
    api_key_id: Optional[str] = None  # Set when approved
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "PendingRegistration":
        return cls(**data)


class AuthManager:
    """
    Authentication manager for ALICE.
    
    Uses JSON file storage for simplicity (no database required).
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the auth manager.
        
        Args:
            data_dir: Directory for storing auth data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.keys_file = self.data_dir / "api_keys.json"
        self.sessions_file = self.data_dir / "sessions.json"
        self.invites_file = self.data_dir / "invites.json"
        self.pending_file = self.data_dir / "pending.json"
        
        self._keys: Dict[str, APIKey] = {}
        self._sessions: Dict[str, Session] = {}
        self._invites: Dict[str, InviteCode] = {}
        self._pending: Dict[str, PendingRegistration] = {}
        self._lock = Lock()
        
        # Load existing data
        self._load()
        
        logger.info("AuthManager initialized: %s", self.data_dir)
    
    def _load(self) -> None:
        """Load data from files."""
        # Load API keys
        if self.keys_file.exists():
            try:
                with open(self.keys_file) as f:
                    data = json.load(f)
                self._keys = {
                    k: APIKey.from_dict(v) for k, v in data.items()
                }
                logger.info("Loaded %d API keys", len(self._keys))
            except Exception as e:
                logger.error("Failed to load API keys: %s", e)
        
        # Load sessions
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file) as f:
                    data = json.load(f)
                self._sessions = {
                    k: Session.from_dict(v) for k, v in data.items()
                }
                # Clean up expired sessions
                self._cleanup_sessions()
                logger.info("Loaded %d sessions", len(self._sessions))
            except Exception as e:
                logger.error("Failed to load sessions: %s", e)
        
        # Load invite codes
        if self.invites_file.exists():
            try:
                with open(self.invites_file) as f:
                    data = json.load(f)
                self._invites = {
                    k: InviteCode.from_dict(v) for k, v in data.items()
                }
                logger.info("Loaded %d invite codes", len(self._invites))
            except Exception as e:
                logger.error("Failed to load invites: %s", e)
        
        # Load pending registrations
        if self.pending_file.exists():
            try:
                with open(self.pending_file) as f:
                    data = json.load(f)
                self._pending = {
                    k: PendingRegistration.from_dict(v) for k, v in data.items()
                }
                logger.info("Loaded %d pending registrations", len(self._pending))
            except Exception as e:
                logger.error("Failed to load pending registrations: %s", e)
    
    def _save_keys(self) -> None:
        """Save API keys to file."""
        try:
            with open(self.keys_file, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._keys.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error("Failed to save API keys: %s", e)
    
    def _save_sessions(self) -> None:
        """Save sessions to file."""
        try:
            with open(self.sessions_file, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._sessions.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error("Failed to save sessions: %s", e)
    
    def _save_invites(self) -> None:
        """Save invite codes to file."""
        try:
            with open(self.invites_file, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._invites.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error("Failed to save invites: %s", e)
    
    def _save_pending(self) -> None:
        """Save pending registrations to file."""
        try:
            with open(self.pending_file, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._pending.items()},
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error("Failed to save pending registrations: %s", e)
    
    def _cleanup_sessions(self) -> None:
        """Remove expired sessions."""
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired()
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            self._save_sessions()
            logger.debug("Cleaned up %d expired sessions", len(expired))
    
    # =========================================================================
    # API KEY MANAGEMENT
    # =========================================================================
    
    def create_api_key(
        self,
        name: str,
        is_admin: bool = False,
        access_level: str = "user",
        rate_limit: int = 100,
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.
        
        Args:
            name: Friendly name for the key
            is_admin: Whether this is an admin key
            access_level: Access level (anonymous, user, admin)
            rate_limit: Requests per minute limit
            
        Returns:
            Tuple of (plaintext_key, APIKey record)
        """
        with self._lock:
            key_id = str(uuid.uuid4())[:8]
            plaintext_key = generate_api_key()
            
            # If is_admin is True, force access_level to admin
            if is_admin:
                access_level = "admin"
            
            api_key = APIKey(
                id=key_id,
                name=name,
                key_hash=hash_key(plaintext_key),
                created_at=time.time(),
                is_admin=is_admin,
                access_level=access_level,
                rate_limit=rate_limit,
            )
            
            self._keys[key_id] = api_key
            self._save_keys()
            
            logger.info("Created API key: %s (%s) with access level: %s", name, key_id, access_level)
            return plaintext_key, api_key
    
    def verify_api_key(self, key: str) -> Optional[APIKey]:
        """
        Verify an API key and return the associated record.
        
        Args:
            key: Plaintext API key
            
        Returns:
            APIKey if valid, None otherwise
        """
        with self._lock:
            for api_key in self._keys.values():
                if api_key.enabled and verify_key(key, api_key.key_hash):
                    # Update last used
                    api_key.last_used = time.time()
                    self._save_keys()
                    return api_key
            return None
    
    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Get an API key by ID."""
        return self._keys.get(key_id)
    
    def list_api_keys(self) -> List[APIKey]:
        """List all API keys (without hashes)."""
        return list(self._keys.values())
    
    def revoke_api_key(self, key_id: str) -> bool:
        """
        Revoke an API key.
        
        Args:
            key_id: Key ID to revoke
            
        Returns:
            True if revoked, False if not found
        """
        with self._lock:
            if key_id in self._keys:
                self._keys[key_id].enabled = False
                self._save_keys()
                logger.info("Revoked API key: %s", key_id)
                return True
            return False
    
    def delete_api_key(self, key_id: str) -> bool:
        """
        Delete an API key.
        
        Args:
            key_id: Key ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key_id in self._keys:
                del self._keys[key_id]
                self._save_keys()
                logger.info("Deleted API key: %s", key_id)
                return True
            return False
    
    # =========================================================================
    # SESSION MANAGEMENT
    # =========================================================================
    
    def create_session(
        self,
        api_key_id: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> Session:
        """
        Create a new session.
        
        Args:
            api_key_id: Associated API key ID (optional)
            duration_seconds: Session duration in seconds (default: 15 minutes)
            metadata: Optional session metadata
            
        Returns:
            New Session
        """
        with self._lock:
            session_id = str(uuid.uuid4())[:12]
            token = secrets.token_urlsafe(32)
            
            # Use provided duration or default 15 minutes
            actual_duration = duration_seconds if duration_seconds is not None else SESSION_TIMEOUT_SECONDS
            
            session = Session(
                id=session_id,
                token=token,
                api_key_id=api_key_id,
                created_at=time.time(),
                last_accessed=time.time(),
                expires_at=time.time() + actual_duration,
                metadata=metadata or {},
            )
            
            self._sessions[session_id] = session
            self._save_sessions()
            
            logger.info("Created session: %s (expires in %d seconds)", session_id, actual_duration)
            return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        session = self._sessions.get(session_id)
        if session and not session.is_expired():
            return session
        return None
    
    def get_session_by_token(self, token: str) -> Optional[Session]:
        """Get a session by token and refresh it."""
        for session in self._sessions.values():
            if session.token == token and not session.is_expired():
                # Refresh the session (update last accessed and extend expiry)
                session.refresh()
                self._save_sessions()
                return session
        return None
    
    def list_sessions(self) -> List[Session]:
        """List all active sessions."""
        self._cleanup_sessions()
        return [s for s in self._sessions.values() if not s.is_expired()]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._save_sessions()
                logger.info("Deleted session: %s", session_id)
                return True
            return False
    
    def update_session_metadata(self, session_id: str, metadata: Dict) -> bool:
        """Update session metadata."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.metadata.update(metadata)
                session.last_accessed = time.time()
                self._save_sessions()
                return True
            return False
    
    # =========================================================================
    # ADMIN OPERATIONS
    # =========================================================================
    
    def get_stats(self) -> Dict:
        """Get authentication statistics."""
        self._cleanup_sessions()
        
        return {
            "total_keys": len(self._keys),
            "active_keys": sum(1 for k in self._keys.values() if k.enabled),
            "admin_keys": sum(1 for k in self._keys.values() if k.is_admin and k.enabled),
            "total_sessions": len(self._sessions),
            "active_sessions": sum(1 for s in self._sessions.values() if not s.is_expired()),
            "active_invites": sum(1 for i in self._invites.values() if i.is_valid()),
            "pending_registrations": sum(1 for p in self._pending.values() if p.status == "pending"),
        }
    
    # =========================================================================
    # INVITE CODE MANAGEMENT
    # =========================================================================
    
    def create_invite(
        self,
        created_by: str,
        uses: int = 1,
        expires_hours: Optional[int] = None,
    ) -> InviteCode:
        """
        Create a new invite code.
        
        Args:
            created_by: Admin key ID that created this invite
            uses: Number of times the invite can be used
            expires_hours: Hours until expiration (None = never)
            
        Returns:
            InviteCode
        """
        with self._lock:
            code = secrets.token_urlsafe(16)
            
            # uses=0 means unlimited, store as -1 internally
            actual_uses = -1 if uses == 0 else uses
            
            invite = InviteCode(
                code=code,
                created_by=created_by,
                created_at=time.time(),
                expires_at=time.time() + (expires_hours * 3600) if expires_hours else None,
                uses_remaining=actual_uses,
                used_by=[],
            )
            
            self._invites[code] = invite
            self._save_invites()
            
            logger.info("Created invite code: %s (uses=%s, expires=%s)", 
                       code[:8] + "...", "unlimited" if actual_uses == -1 else actual_uses, expires_hours)
            return invite
    
    def verify_invite(self, code: str) -> Optional[InviteCode]:
        """
        Verify an invite code is valid.
        
        Args:
            code: Invite code
            
        Returns:
            InviteCode if valid, None otherwise
        """
        invite = self._invites.get(code)
        if invite and invite.is_valid():
            return invite
        return None
    
    def use_invite(self, code: str, key_id: str) -> bool:
        """
        Use an invite code (decrement uses).
        
        Args:
            code: Invite code
            key_id: API key ID created with this invite
            
        Returns:
            True if successful
        """
        with self._lock:
            invite = self._invites.get(code)
            if invite and invite.is_valid():
                # Only decrement if not unlimited (-1 means unlimited)
                if invite.uses_remaining > 0:
                    invite.uses_remaining -= 1
                invite.used_by.append(key_id)
                self._save_invites()
                return True
            return False
    
    def list_invites(self) -> List[InviteCode]:
        """List all invite codes."""
        return list(self._invites.values())
    
    def delete_invite(self, code: str) -> bool:
        """Delete an invite code."""
        with self._lock:
            if code in self._invites:
                del self._invites[code]
                self._save_invites()
                return True
            return False
    
    # =========================================================================
    # PENDING REGISTRATION MANAGEMENT
    # =========================================================================
    
    def create_pending_registration(self, name: str) -> PendingRegistration:
        """
        Create a pending registration request.
        
        Args:
            name: Requested account name
            
        Returns:
            PendingRegistration
        """
        with self._lock:
            reg_id = str(uuid.uuid4())[:8]
            
            pending = PendingRegistration(
                id=reg_id,
                name=name,
                created_at=time.time(),
                status="pending",
            )
            
            self._pending[reg_id] = pending
            self._save_pending()
            
            logger.info("Created pending registration: %s (%s)", name, reg_id)
            return pending
    
    def approve_registration(
        self,
        reg_id: str,
        approved_by: str,
    ) -> Optional[tuple[str, APIKey]]:
        """
        Approve a pending registration and create API key.
        
        Args:
            reg_id: Registration ID
            approved_by: Admin key ID that approved
            
        Returns:
            Tuple of (plaintext_key, APIKey) if successful
        """
        with self._lock:
            pending = self._pending.get(reg_id)
            if not pending or pending.status != "pending":
                return None
            
            # Create the API key
            plaintext_key, api_key = self.create_api_key(
                name=pending.name,
                is_admin=False,
                access_level="user",
            )
            
            # Update pending record
            pending.status = "approved"
            pending.approved_by = approved_by
            pending.approved_at = time.time()
            pending.api_key_id = api_key.id
            self._save_pending()
            
            logger.info("Approved registration: %s -> %s", reg_id, api_key.id)
            return plaintext_key, api_key
    
    def reject_registration(self, reg_id: str, rejected_by: str) -> bool:
        """
        Reject a pending registration.
        
        Args:
            reg_id: Registration ID
            rejected_by: Admin key ID that rejected
            
        Returns:
            True if successful
        """
        with self._lock:
            pending = self._pending.get(reg_id)
            if not pending or pending.status != "pending":
                return False
            
            pending.status = "rejected"
            pending.approved_by = rejected_by  # Reuse field for rejector
            pending.approved_at = time.time()
            self._save_pending()
            
            logger.info("Rejected registration: %s", reg_id)
            return True
    
    def list_pending_registrations(self, status: Optional[str] = None) -> List[PendingRegistration]:
        """List pending registrations, optionally filtered by status."""
        if status:
            return [p for p in self._pending.values() if p.status == status]
        return list(self._pending.values())
    
    def delete_pending_registration(self, reg_id: str) -> bool:
        """Delete a pending registration."""
        with self._lock:
            if reg_id in self._pending:
                del self._pending[reg_id]
                self._save_pending()
                return True
            return False
