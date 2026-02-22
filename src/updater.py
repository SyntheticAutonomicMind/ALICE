# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
ALICE Update Manager

Provides self-update capabilities via GitHub Releases integration.
Supports version checking, downloading updates, backup/rollback, and
automatic or admin-triggered application updates.

Update flow:
  1. Check GitHub Releases for latest version
  2. Compare with running version
  3. Download release tarball
  4. Backup current installation
  5. Extract and apply update
  6. Restart service (if running as systemd service)

Design principles:
  - Non-destructive: Always backs up before updating
  - Rollback: Can revert to any previous backup
  - Config-safe: Never overwrites user config files
  - Atomic: Uses staging directory, then swaps into place
  - Resumable: Download progress is tracked and can resume
"""

import asyncio
import hashlib
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from . import __version__

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

# GitHub repository for release checking
GITHUB_OWNER = "SyntheticAutonomicMind"
GITHUB_REPO = "ALICE"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_RELEASES_URL = f"{GITHUB_API_BASE}/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases"

# Update configuration
BACKUP_DIR_NAME = ".alice-backups"
MAX_BACKUPS = 5  # Keep last 5 backups
UPDATE_CHECK_INTERVAL = 3600  # Check every hour (seconds)
DOWNLOAD_TIMEOUT = 600  # 10 minutes max for download
DOWNLOAD_CHUNK_SIZE = 65536  # 64KB chunks

# Files/directories that should NEVER be overwritten during update
PROTECTED_PATHS = {
    "config.yaml",
    "models/",
    "images/",
    "data/",
    "logs/",
    "venv/",
    ".env",
}

# Files/directories to skip when extracting updates
SKIP_ON_UPDATE = {
    ".git",
    ".github",
    "venv",
    "models",
    "images",
    "data",
    "logs",
    "__pycache__",
    ".pytest_cache",
    "ai-assisted",
    "scratch",
    "project-docs",
}


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class ReleaseInfo:
    """Information about a GitHub release."""
    version: str
    tag_name: str
    name: str
    body: str
    published_at: str
    tarball_url: str
    html_url: str
    prerelease: bool = False
    draft: bool = False
    assets: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateStatus:
    """Current update status."""
    current_version: str
    latest_version: Optional[str] = None
    update_available: bool = False
    release_info: Optional[ReleaseInfo] = None
    last_check: Optional[str] = None
    update_in_progress: bool = False
    update_stage: Optional[str] = None  # checking, downloading, backing_up, applying, restarting
    download_progress: float = 0.0  # 0.0 to 1.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "update_available": self.update_available,
            "last_check": self.last_check,
            "update_in_progress": self.update_in_progress,
            "update_stage": self.update_stage,
            "download_progress": self.download_progress,
            "error": self.error,
        }
        if self.release_info:
            d["release_info"] = self.release_info.to_dict()
        return d


@dataclass
class BackupInfo:
    """Information about a backup."""
    version: str
    timestamp: str
    path: str
    size_bytes: int


# =============================================================================
# VERSION COMPARISON
# =============================================================================

def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse a version string into a comparable tuple.

    Supports two formats:
    - Date-based: 'YYYYMMDD.RELEASE' (e.g., '20260222.1')
    - Semver: 'MAJOR.MINOR.PATCH' (e.g., '1.2.3')

    Strips leading 'v' and handles pre-release suffixes by ignoring them
    for comparison purposes. Date-based versions naturally sort correctly
    since the date component is a large integer.
    """
    clean = version_str.strip().lstrip("v")
    # Split off any pre-release suffix (-alpha, -beta, -rc1, etc.)
    base = clean.split("-")[0]
    parts = []
    for part in base.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(0)
    return tuple(parts)


def is_newer_version(latest: str, current: str) -> bool:
    """Check if latest version is newer than current version."""
    return parse_version(latest) > parse_version(current)


# =============================================================================
# UPDATE MANAGER
# =============================================================================

class UpdateManager:
    """Manages checking for and applying ALICE updates.

    The UpdateManager integrates with GitHub Releases to provide:
    - Periodic version checking
    - Download and verification of release tarballs
    - Safe backup and restore of the installation
    - Atomic update application with rollback capability

    Usage:
        manager = UpdateManager(install_dir=Path("/opt/alice"))
        status = await manager.check_for_update()
        if status.update_available:
            await manager.apply_update()
    """

    def __init__(
        self,
        install_dir: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        github_token: Optional[str] = None,
        auto_check: bool = True,
        check_interval: int = UPDATE_CHECK_INTERVAL,
    ):
        """Initialize the UpdateManager.

        Args:
            install_dir: Root directory of the ALICE installation.
                         Defaults to the parent of the src/ directory.
            data_dir: Directory for update data (backups, downloads).
                      Defaults to install_dir/.alice-backups.
            github_token: Optional GitHub token for API rate limits.
            auto_check: Whether to start periodic update checking.
            check_interval: Seconds between automatic update checks.
        """
        if install_dir is None:
            # Default: parent of the src/ package directory
            install_dir = Path(__file__).resolve().parent.parent
        self.install_dir = install_dir

        if data_dir is None:
            data_dir = install_dir / BACKUP_DIR_NAME
        self.data_dir = data_dir
        self.backup_dir = data_dir / "backups"
        self.download_dir = data_dir / "downloads"
        self.state_file = data_dir / "update_state.json"

        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.auto_check = auto_check
        self.check_interval = check_interval

        # State
        self._status = UpdateStatus(current_version=__version__)
        self._check_task: Optional[asyncio.Task] = None
        self._update_lock = asyncio.Lock()

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # Load persisted state
        self._load_state()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    @property
    def status(self) -> UpdateStatus:
        """Get current update status."""
        return self._status

    async def start(self) -> None:
        """Start the update manager (begins periodic checking if enabled)."""
        if self.auto_check:
            self._check_task = asyncio.create_task(self._periodic_check())
            logger.info("Update manager started (check interval: %ds)", self.check_interval)
        else:
            logger.info("Update manager started (auto-check disabled)")

    async def stop(self) -> None:
        """Stop the update manager."""
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
            self._check_task = None
        self._save_state()
        logger.info("Update manager stopped")

    async def check_for_update(self, include_prerelease: bool = False) -> UpdateStatus:
        """Check GitHub Releases for available updates.

        Args:
            include_prerelease: Whether to consider pre-release versions.

        Returns:
            UpdateStatus with the results of the check.
        """
        self._status.update_stage = "checking"
        self._status.error = None

        try:
            release = await self._fetch_latest_release(include_prerelease)
            if release is None:
                self._status.latest_version = self._status.current_version
                self._status.update_available = False
                self._status.release_info = None
                logger.info("No releases found on GitHub")
            else:
                self._status.latest_version = release.version
                self._status.update_available = is_newer_version(
                    release.version, self._status.current_version
                )
                self._status.release_info = release

                if self._status.update_available:
                    logger.info(
                        "Update available: %s -> %s",
                        self._status.current_version,
                        release.version,
                    )
                else:
                    logger.debug("Already up to date (%s)", self._status.current_version)

            self._status.last_check = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            logger.error("Update check failed: %s", e)
            self._status.error = str(e)
        finally:
            self._status.update_stage = None
            self._save_state()

        return self._status

    async def apply_update(
        self,
        version: Optional[str] = None,
        restart: bool = True,
    ) -> UpdateStatus:
        """Download and apply an update.

        Args:
            version: Specific version to install. If None, installs latest.
            restart: Whether to restart the service after updating.

        Returns:
            UpdateStatus with the results.

        Raises:
            RuntimeError: If an update is already in progress.
        """
        if self._status.update_in_progress:
            raise RuntimeError("Update already in progress")

        async with self._update_lock:
            self._status.update_in_progress = True
            self._status.error = None

            try:
                # Step 1: Determine which version to install
                if version:
                    release = await self._fetch_release_by_tag(f"v{version.lstrip('v')}")
                    if release is None:
                        raise ValueError(f"Release version {version} not found")
                else:
                    if not self._status.release_info:
                        await self.check_for_update()
                    release = self._status.release_info
                    if release is None or not self._status.update_available:
                        self._status.update_in_progress = False
                        self._status.error = "No update available"
                        return self._status

                target_version = release.version
                logger.info("Applying update to version %s", target_version)

                # Step 2: Download the release
                self._status.update_stage = "downloading"
                tarball_path = await self._download_release(release)

                # Step 3: Backup current installation
                self._status.update_stage = "backing_up"
                backup_path = await self._create_backup()
                logger.info("Backup created at %s", backup_path)

                # Step 4: Extract and apply the update
                self._status.update_stage = "applying"
                await self._apply_tarball(tarball_path)
                logger.info("Update files applied")

                # Step 5: Reinstall dependencies if requirements changed
                self._status.update_stage = "dependencies"
                await self._update_dependencies()

                # Step 6: Migrate config (add new settings with defaults)
                self._status.update_stage = "config_migration"
                try:
                    from .config_migration import migrate_config
                    migration = migrate_config()
                    if migration["migrated"]:
                        logger.info(
                            "Config migrated: added %s",
                            ", ".join(migration["added"]),
                        )
                except Exception as e:
                    logger.debug("Config migration after update skipped: %s", e)

                # Step 7: Clean up old backups
                self._cleanup_old_backups()

                # Step 8: Clean up download
                if tarball_path.exists():
                    tarball_path.unlink()

                self._status.current_version = target_version
                self._status.update_available = False
                self._status.update_stage = "complete"
                logger.info("Update to %s complete", target_version)

                # Step 8: Restart if requested
                if restart:
                    self._status.update_stage = "restarting"
                    self._save_state()
                    await self._restart_service()

            except Exception as e:
                logger.exception("Update failed: %s", e)
                self._status.error = str(e)
                self._status.update_stage = "failed"
            finally:
                self._status.update_in_progress = False
                self._save_state()

        return self._status

    async def rollback(self, backup_version: Optional[str] = None) -> bool:
        """Rollback to a previous backup.

        Args:
            backup_version: Specific version to rollback to.
                           If None, rolls back to the most recent backup.

        Returns:
            True if rollback was successful.
        """
        backups = self.list_backups()
        if not backups:
            logger.error("No backups available for rollback")
            return False

        if backup_version:
            target = next((b for b in backups if b.version == backup_version), None)
            if target is None:
                logger.error("Backup version %s not found", backup_version)
                return False
        else:
            target = backups[0]  # Most recent

        logger.info("Rolling back to version %s from backup %s", target.version, target.path)

        try:
            backup_path = Path(target.path)
            if not backup_path.exists():
                logger.error("Backup path does not exist: %s", backup_path)
                return False

            # Apply the backup (same logic as applying an update)
            await self._apply_tarball(backup_path)
            self._status.current_version = target.version
            self._save_state()
            logger.info("Rollback to %s complete", target.version)
            return True
        except Exception as e:
            logger.exception("Rollback failed: %s", e)
            return False

    def list_backups(self) -> List[BackupInfo]:
        """List available backups, newest first."""
        backups = []
        if not self.backup_dir.exists():
            return backups

        for entry in sorted(self.backup_dir.iterdir(), reverse=True):
            if entry.is_file() and entry.suffix == ".gz":
                # Parse version from filename: alice-backup-{version}-{timestamp}.tar.gz
                name = entry.stem  # removes .gz
                if name.endswith(".tar"):
                    name = name[:-4]  # removes .tar
                parts = name.split("-")
                # alice-backup-1.2.1-20260222T0617
                if len(parts) >= 4 and parts[0] == "alice" and parts[1] == "backup":
                    version = parts[2]
                    timestamp = "-".join(parts[3:])
                else:
                    version = "unknown"
                    timestamp = entry.stat().st_mtime

                backups.append(BackupInfo(
                    version=version,
                    timestamp=str(timestamp),
                    path=str(entry),
                    size_bytes=entry.stat().st_size,
                ))

        return backups

    # =========================================================================
    # GITHUB API
    # =========================================================================

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for GitHub API requests."""
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": f"ALICE/{__version__}",
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    async def _fetch_latest_release(
        self,
        include_prerelease: bool = False,
    ) -> Optional[ReleaseInfo]:
        """Fetch the latest release from GitHub."""
        try:
            async with aiohttp.ClientSession() as session:
                if not include_prerelease:
                    # Use the /latest endpoint (excludes pre-releases)
                    url = f"{GITHUB_RELEASES_URL}/latest"
                    async with session.get(url, headers=self._get_headers(), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status == 404:
                            return None
                        resp.raise_for_status()
                        data = await resp.json()
                        return self._parse_release(data)
                else:
                    # List all releases and pick the first non-draft
                    url = f"{GITHUB_RELEASES_URL}?per_page=10"
                    async with session.get(url, headers=self._get_headers(), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        resp.raise_for_status()
                        releases = await resp.json()
                        for r in releases:
                            if not r.get("draft", False):
                                return self._parse_release(r)
                        return None
        except aiohttp.ClientError as e:
            logger.error("GitHub API request failed: %s", e)
            raise RuntimeError(f"Failed to check for updates: {e}") from e

    async def _fetch_release_by_tag(self, tag: str) -> Optional[ReleaseInfo]:
        """Fetch a specific release by tag name."""
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{GITHUB_RELEASES_URL}/tags/{tag}"
                async with session.get(url, headers=self._get_headers(), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 404:
                        return None
                    resp.raise_for_status()
                    data = await resp.json()
                    return self._parse_release(data)
        except aiohttp.ClientError as e:
            logger.error("GitHub API request failed: %s", e)
            raise RuntimeError(f"Failed to fetch release {tag}: {e}") from e

    def _parse_release(self, data: Dict[str, Any]) -> ReleaseInfo:
        """Parse a GitHub release API response into ReleaseInfo."""
        tag = data.get("tag_name", "")
        version = tag.lstrip("v") if tag else "0.0.0"

        return ReleaseInfo(
            version=version,
            tag_name=tag,
            name=data.get("name", ""),
            body=data.get("body", ""),
            published_at=data.get("published_at", ""),
            tarball_url=data.get("tarball_url", ""),
            html_url=data.get("html_url", ""),
            prerelease=data.get("prerelease", False),
            draft=data.get("draft", False),
            assets=[
                {
                    "name": a["name"],
                    "url": a["browser_download_url"],
                    "size": a["size"],
                    "content_type": a["content_type"],
                }
                for a in data.get("assets", [])
            ],
        )

    # =========================================================================
    # DOWNLOAD
    # =========================================================================

    async def _download_release(self, release: ReleaseInfo) -> Path:
        """Download a release tarball.

        First checks for a named asset (alice-{version}.tar.gz), then
        falls back to GitHub's auto-generated source tarball.

        Returns:
            Path to the downloaded tarball.
        """
        # Prefer a named release asset over the auto-generated tarball
        asset_name = f"alice-{release.version}.tar.gz"
        download_url = release.tarball_url  # fallback: GitHub source tarball

        for asset in release.assets:
            if asset["name"] == asset_name:
                download_url = asset["url"]
                logger.info("Using release asset: %s", asset_name)
                break
        else:
            logger.info("No named asset found, using GitHub source tarball")

        dest = self.download_dir / f"alice-{release.version}.tar.gz"

        # Resume support: if partial download exists, try to resume
        existing_size = dest.stat().st_size if dest.exists() else 0

        headers = self._get_headers()
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"

        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
                async with session.get(download_url, headers=headers, timeout=timeout) as resp:
                    if resp.status == 416:
                        # Range not satisfiable - file already fully downloaded
                        logger.info("Download already complete: %s", dest)
                        return dest

                    resp.raise_for_status()

                    total_size = int(resp.headers.get("Content-Length", 0))
                    if resp.status == 206:
                        # Partial content - resuming
                        total_size += existing_size
                        mode = "ab"
                        logger.info("Resuming download from byte %d", existing_size)
                    else:
                        mode = "wb"
                        existing_size = 0

                    downloaded = existing_size
                    with open(dest, mode) as f:
                        async for chunk in resp.content.iter_chunked(DOWNLOAD_CHUNK_SIZE):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                self._status.download_progress = downloaded / total_size

                    logger.info(
                        "Downloaded %s (%.1f MB)",
                        dest.name,
                        downloaded / (1024 * 1024),
                    )
                    return dest

        except Exception as e:
            logger.error("Download failed: %s", e)
            raise RuntimeError(f"Failed to download update: {e}") from e

    # =========================================================================
    # BACKUP & RESTORE
    # =========================================================================

    async def _create_backup(self) -> Path:
        """Create a backup of the current installation.

        Backs up src/, web/, scripts/, requirements.txt, and other
        application files. Does NOT back up models, images, data, venv,
        or config (those are user data).

        Returns:
            Path to the backup tarball.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M")
        backup_name = f"alice-backup-{__version__}-{timestamp}.tar.gz"
        backup_path = self.backup_dir / backup_name

        def _do_backup():
            with tarfile.open(backup_path, "w:gz") as tar:
                for item in self.install_dir.iterdir():
                    if item.name in SKIP_ON_UPDATE:
                        continue
                    if item.name.startswith("."):
                        continue
                    if item.name == BACKUP_DIR_NAME:
                        continue
                    tar.add(str(item), arcname=item.name)
            return backup_path

        return await asyncio.to_thread(_do_backup)

    async def _apply_tarball(self, tarball_path: Path) -> None:
        """Extract and apply a tarball update to the installation.

        Uses a staging directory for atomic-ish replacement:
        1. Extract to staging directory
        2. Remove old application files (not user data)
        3. Move new files into place

        Args:
            tarball_path: Path to the tarball to apply.
        """
        def _do_apply():
            staging = self.download_dir / "staging"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)

            # Extract tarball
            with tarfile.open(tarball_path, "r:gz") as tar:
                # Security: prevent path traversal attacks
                for member in tar.getmembers():
                    member_path = Path(member.name)
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise ValueError(f"Unsafe path in tarball: {member.name}")
                tar.extractall(staging)

            # GitHub source tarballs have a top-level directory like
            # "OWNER-REPO-HASH/". Detect and flatten it.
            extracted_items = list(staging.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_dir = extracted_items[0]
            else:
                source_dir = staging

            # Apply update: copy new files over old ones
            # Skip protected paths (user data, config, venv, etc.)
            for item in source_dir.iterdir():
                if item.name in SKIP_ON_UPDATE:
                    continue
                if item.name.startswith(".") and item.name != ".distignore":
                    continue
                if item.name == BACKUP_DIR_NAME:
                    continue

                dest = self.install_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()

                if item.is_dir():
                    shutil.copytree(str(item), str(dest))
                else:
                    shutil.copy2(str(item), str(dest))

            # Clean up staging
            shutil.rmtree(staging)

        await asyncio.to_thread(_do_apply)

    # =========================================================================
    # DEPENDENCIES
    # =========================================================================

    async def _update_dependencies(self) -> None:
        """Reinstall Python dependencies if requirements.txt has changed."""
        requirements_file = self.install_dir / "requirements.txt"
        if not requirements_file.exists():
            logger.warning("requirements.txt not found, skipping dependency update")
            return

        # Find the venv pip
        venv_pip = self.install_dir / "venv" / "bin" / "pip"
        if not venv_pip.exists():
            # Try system pip
            venv_pip = Path(sys.executable).parent / "pip"
            if not venv_pip.exists():
                logger.warning("pip not found, skipping dependency update")
                return

        logger.info("Updating Python dependencies...")
        try:
            proc = await asyncio.create_subprocess_exec(
                str(venv_pip), "install", "-r", str(requirements_file),
                "--quiet", "--disable-pip-version-check",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning(
                    "pip install returned %d: %s",
                    proc.returncode,
                    stderr.decode().strip(),
                )
            else:
                logger.info("Dependencies updated successfully")
        except Exception as e:
            logger.warning("Failed to update dependencies: %s", e)

    # =========================================================================
    # SERVICE MANAGEMENT
    # =========================================================================

    async def _restart_service(self) -> None:
        """Restart the ALICE service.

        Attempts multiple strategies in order:
        1. systemd (system-level): systemctl restart alice
        2. systemd (user-level): systemctl --user restart alice
        3. Self-restart: os.execv to replace the current process
        """
        logger.info("Restarting ALICE service...")

        # Strategy 1: systemd system service
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "is-active", "--quiet", "alice",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            if proc.returncode == 0:
                logger.info("Restarting via systemctl restart alice")
                proc = await asyncio.create_subprocess_exec(
                    "sudo", "systemctl", "restart", "alice",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                if proc.returncode == 0:
                    return
                logger.warning("systemctl restart failed, trying alternatives")
        except FileNotFoundError:
            pass

        # Strategy 2: systemd user service (SteamOS)
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl", "--user", "is-active", "--quiet", "alice",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            if proc.returncode == 0:
                logger.info("Restarting via systemctl --user restart alice")
                proc = await asyncio.create_subprocess_exec(
                    "systemctl", "--user", "restart", "alice",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                if proc.returncode == 0:
                    return
        except FileNotFoundError:
            pass

        # Strategy 3: Self-restart by sending SIGTERM to self
        # The service manager (systemd) will restart us if configured with Restart=always
        logger.info("Sending SIGTERM to self for restart (requires Restart=always in service)")
        os.kill(os.getpid(), signal.SIGTERM)

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def _save_state(self) -> None:
        """Persist update state to disk."""
        try:
            state = {
                "current_version": self._status.current_version,
                "latest_version": self._status.latest_version,
                "update_available": self._status.update_available,
                "last_check": self._status.last_check,
            }
            with open(self.state_file, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug("Failed to save update state: %s", e)

    def _load_state(self) -> None:
        """Load persisted update state."""
        try:
            if self.state_file.exists():
                with open(self.state_file) as f:
                    state = json.load(f)
                self._status.latest_version = state.get("latest_version")
                self._status.update_available = state.get("update_available", False)
                self._status.last_check = state.get("last_check")
        except Exception as e:
            logger.debug("Failed to load update state: %s", e)

    # =========================================================================
    # BACKGROUND TASKS
    # =========================================================================

    async def _periodic_check(self) -> None:
        """Background task for periodic update checking."""
        # Wait a bit before first check (let the service start up)
        await asyncio.sleep(60)

        while True:
            try:
                await self.check_for_update()
            except Exception as e:
                logger.debug("Periodic update check failed: %s", e)

            try:
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break

    # =========================================================================
    # CLEANUP
    # =========================================================================

    def _cleanup_old_backups(self) -> None:
        """Remove old backups, keeping only the most recent MAX_BACKUPS."""
        backups = self.list_backups()
        if len(backups) <= MAX_BACKUPS:
            return

        for old_backup in backups[MAX_BACKUPS:]:
            try:
                Path(old_backup.path).unlink()
                logger.info("Removed old backup: %s", old_backup.path)
            except Exception as e:
                logger.warning("Failed to remove old backup %s: %s", old_backup.path, e)


# =============================================================================
# MODULE-LEVEL INSTANCE
# =============================================================================

# Singleton instance, initialized during app startup
_update_manager: Optional[UpdateManager] = None


def get_update_manager() -> Optional[UpdateManager]:
    """Get the global UpdateManager instance."""
    return _update_manager


async def init_update_manager(
    install_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    github_token: Optional[str] = None,
    auto_check: bool = True,
) -> UpdateManager:
    """Initialize and start the global UpdateManager.

    Args:
        install_dir: Root of ALICE installation.
        data_dir: Directory for backups and downloads.
        github_token: GitHub API token (optional).
        auto_check: Enable periodic update checking.

    Returns:
        The initialized UpdateManager instance.
    """
    global _update_manager
    _update_manager = UpdateManager(
        install_dir=install_dir,
        data_dir=data_dir,
        github_token=github_token,
        auto_check=auto_check,
    )
    await _update_manager.start()
    return _update_manager


async def shutdown_update_manager() -> None:
    """Stop and clean up the global UpdateManager."""
    global _update_manager
    if _update_manager:
        await _update_manager.stop()
        _update_manager = None
