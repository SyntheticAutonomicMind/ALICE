# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Model Catalog Cache

Caches complete model catalogs from CivitAI and HuggingFace for fast local search.
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SyncStatus:
    """Status of catalog synchronization."""
    last_sync_civitai: Optional[datetime]
    last_sync_huggingface: Optional[datetime]
    civitai_models: int
    huggingface_models: int
    sync_in_progress: bool
    current_page: Optional[int]
    total_pages: Optional[int]
    error: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "last_sync_civitai": self.last_sync_civitai.isoformat() if self.last_sync_civitai else None,
            "last_sync_huggingface": self.last_sync_huggingface.isoformat() if self.last_sync_huggingface else None,
            "civitai_models": self.civitai_models,
            "huggingface_models": self.huggingface_models,
            "sync_in_progress": self.sync_in_progress,
            "current_page": self.current_page,
            "total_pages": self.total_pages,
            "error": self.error,
        }


class ModelCacheService:
    """
    Service for caching and searching model catalogs.
    
    Downloads complete catalogs from CivitAI and HuggingFace APIs,
    stores them in SQLite, and provides fast local search.
    """
    
    CIVITAI_API_BASE = "https://civitai.com/api/v1"
    HUGGINGFACE_API_BASE = "https://huggingface.co/api"
    
    def __init__(
        self,
        database_path: Path,
        civitai_api_key: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        civitai_page_limit: Optional[int] = None,
        huggingface_limit: int = 10000,
    ):
        """
        Initialize the model cache service.
        
        Args:
            database_path: Path to SQLite database file
            civitai_api_key: Optional CivitAI API key for authentication
            huggingface_token: Optional HuggingFace token
            civitai_page_limit: Max pages to fetch from CivitAI (None = all)
            huggingface_limit: Max models to fetch from HuggingFace
        """
        self.database_path = Path(database_path)
        self.civitai_api_key = civitai_api_key
        self.huggingface_token = huggingface_token
        self.civitai_page_limit = civitai_page_limit
        self.huggingface_limit = huggingface_limit
        
        # Sync state
        self._sync_in_progress = False
        self._current_page = None
        self._total_pages = None
        self._sync_error = None
        self._has_fts5 = False  # Will be set during database init
        
        # Initialize database
        self._init_database()
        
        logger.info(
            "ModelCacheService initialized: database=%s, civitai_limit=%s, hf_limit=%d",
            self.database_path, civitai_page_limit or "unlimited", huggingface_limit
        )
    
    def _init_database(self) -> None:
        """Initialize the SQLite database schema."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(self.database_path))
        cursor = conn.cursor()
        
        try:
            # Create models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT,
                    base_model TEXT,
                    creator TEXT,
                    description TEXT,
                    tags TEXT,
                    nsfw INTEGER,
                    download_url TEXT,
                    file_size INTEGER,
                    thumbnail_url TEXT,
                    rating REAL,
                    download_count INTEGER,
                    created_at TEXT,
                    updated_at TEXT,
                    metadata TEXT,
                    last_synced TEXT
                )
            """)
            
            # Create indexes for fast querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON models(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON models(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_base_model ON models(base_model)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_creator ON models(creator)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nsfw ON models(nsfw)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON models(name)")
            
            # Create sync metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sync_metadata (
                    source TEXT PRIMARY KEY,
                    last_sync TEXT,
                    record_count INTEGER,
                    error TEXT
                )
            """)
            
            # Commit core tables first
            conn.commit()
            
            # Try to create full-text search table (FTS5) if available
            # This is optional - degrades gracefully if SQLite doesn't have FTS5 compiled in
            self._has_fts5 = False
            try:
                cursor.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS models_fts USING fts5(
                        id UNINDEXED,
                        name,
                        description,
                        tags,
                        creator,
                        content='models',
                        content_rowid='rowid'
                    )
                """)
                
                # Create triggers to keep FTS in sync
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS models_ai AFTER INSERT ON models BEGIN
                        INSERT INTO models_fts(rowid, id, name, description, tags, creator)
                        VALUES (new.rowid, new.id, new.name, new.description, new.tags, new.creator);
                    END
                """)
                
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS models_ad AFTER DELETE ON models BEGIN
                        DELETE FROM models_fts WHERE rowid = old.rowid;
                    END
                """)
                
                cursor.execute("""
                    CREATE TRIGGER IF NOT EXISTS models_au AFTER UPDATE ON models BEGIN
                        DELETE FROM models_fts WHERE rowid = old.rowid;
                        INSERT INTO models_fts(rowid, id, name, description, tags, creator)
                        VALUES (new.rowid, new.id, new.name, new.description, new.tags, new.creator);
                    END
                """)
                
                conn.commit()
                self._has_fts5 = True
                logger.info("FTS5 full-text search enabled")
                
            except sqlite3.OperationalError as e:
                if "no such module: fts5" in str(e):
                    logger.warning("FTS5 not available in SQLite, using basic search (slower)")
                else:
                    logger.error("Error creating FTS5 table: %s", e)
                    # Don't raise - FTS5 is optional
            
            logger.info("Database schema initialized")
            
        finally:
            conn.close()
    
    async def sync_civitai(self, progress_callback: Optional[callable] = None) -> None:
        """
        Synchronize CivitAI model catalog.
        
        Fetches all models via pagination and stores in database.
        
        Args:
            progress_callback: Optional callback(current_page, total_pages)
        """
        if self._sync_in_progress:
            logger.warning("CivitAI sync already in progress")
            return
        
        self._sync_in_progress = True
        self._sync_error = None
        
        try:
            logger.info("Starting CivitAI catalog sync...")
            
            cursor = None  # Start with no cursor (first page)
            total_models = 0
            page_count = 0
            
            async with aiohttp.ClientSession() as session:
                while True:
                    # Check page limit
                    if self.civitai_page_limit and page_count >= self.civitai_page_limit:
                        logger.info("Reached page limit %d, stopping", self.civitai_page_limit)
                        break
                    
                    try:
                        page_count += 1
                        
                        # Fetch page using cursor (or page 1 if no cursor yet)
                        params = {
                            "limit": 100,
                            "sort": "Highest Rated",
                            "nsfw": "true",
                        }
                        
                        if cursor:
                            # Use cursor for pagination (required after page ~20)
                            params["cursor"] = cursor
                            logger.info("Fetching CivitAI page %d with cursor: %s...", page_count, cursor[:50])
                        else:
                            # First page uses page=1
                            params["page"] = 1
                            logger.info("Fetching CivitAI page 1 (initial request)")
                        
                        # CivitAI uses query parameter for authentication, not header
                        if self.civitai_api_key:
                            params["token"] = self.civitai_api_key
                        # CivitAI uses query parameter for authentication, not header
                        if self.civitai_api_key:
                            params["token"] = self.civitai_api_key
                        else:
                            logger.warning("No CivitAI API key configured - may be rate limited")
                        
                        self._current_page = page_count
                        
                        async with session.get(
                            f"{self.CIVITAI_API_BASE}/models",
                            params=params,
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as response:
                            logger.info("CivitAI response status: %d", response.status)
                            if response.status == 429:
                                # Rate limited - wait and retry (cap at 10 seconds)
                                retry_after = min(int(response.headers.get("Retry-After", "10")), 10)
                                logger.warning("Rate limited (HTTP 429), waiting %d seconds (API requested %s)", 
                                             retry_after, response.headers.get("Retry-After", "unknown"))
                                await asyncio.sleep(retry_after)
                                continue
                            
                            if response.status != 200:
                                error_text = await response.text()
                                logger.error("CivitAI API error: %s - %s", response.status, error_text[:200])
                                break
                            
                            data = await response.json()
                        
                        # Get next cursor from metadata
                        metadata = data.get("metadata", {})
                        next_cursor = metadata.get("nextCursor")
                        self._total_pages = metadata.get("totalPages")  # May not be accurate with cursors
                        
                        if self._total_pages:
                            logger.info("Progress: page %d of %d", page_count, self._total_pages)
                        else:
                            logger.info("Progress: page %d (total unknown)", page_count)
                        
                        if self._total_pages:
                            logger.info("Progress: page %d of %d", page_count, self._total_pages)
                        else:
                            logger.info("Progress: page %d (total unknown)", page_count)
                        
                        if progress_callback:
                            progress_callback(page_count, self._total_pages)
                        
                        # Process models from this page
                        items = data.get("items", [])
                        if not items:
                            logger.info("No more models, stopping at page %d", page_count)
                            break
                        
                        # Store models in database
                        models_data = []
                        for item in items:
                            try:
                                model_data = self._parse_civitai_model(item)
                                if model_data:
                                    models_data.append(model_data)
                            except Exception as e:
                                logger.debug("Error parsing model: %s", e)
                                continue
                        
                        if models_data:
                            self._bulk_insert_models(models_data)
                            total_models += len(models_data)
                            logger.info(
                                "Stored %d models from page %d (total: %d)",
                                len(models_data), page_count, total_models
                            )
                        
                        # Check if there's a next cursor
                        if not next_cursor:
                            logger.info("No more pages (nextCursor is null), stopping at page %d", page_count)
                            break
                        
                        # Use the next cursor for the next iteration
                        cursor = next_cursor
                        # Use the next cursor for the next iteration
                        cursor = next_cursor
                        
                        # Rate limiting - be nice to the API (2 seconds between requests)
                        await asyncio.sleep(2.0)
                        
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error("Error fetching page %d: %s", page_count, e)
                        self._sync_error = str(e)
                        # Continue to next page on error if we have a cursor
                        if cursor:
                            logger.info("Attempting to continue with next cursor...")
                        else:
                            break
                        await asyncio.sleep(2.0)
            
            # Update sync metadata
            self._update_sync_metadata("civitai", total_models)
            
            logger.info("CivitAI sync complete: %d models synced", total_models)
            
        except asyncio.CancelledError:
            logger.warning("CivitAI sync cancelled")
            raise
        except Exception as e:
            logger.error("CivitAI sync failed: %s", e)
            self._sync_error = str(e)
        finally:
            self._sync_in_progress = False
            self._current_page = None
            self._total_pages = None
    
    async def sync_huggingface(self, progress_callback: Optional[callable] = None) -> None:
        """
        Synchronize HuggingFace model catalog.
        
        Fetches Stable Diffusion models and stores in database.
        
        Args:
            progress_callback: Optional callback(current_count, total)
        """
        if self._sync_in_progress:
            logger.warning("HuggingFace sync already in progress")
            return
        
        self._sync_in_progress = True
        self._sync_error = None
        
        try:
            logger.info("Starting HuggingFace catalog sync...")
            
            headers = {}
            if self.huggingface_token:
                headers["Authorization"] = f"Bearer {self.huggingface_token}"
            
            total_models = 0
            skip = 0
            limit = 100
            
            async with aiohttp.ClientSession() as session:
                while skip < self.huggingface_limit:
                    try:
                        # Fetch batch
                        params = {
                            "filter": "diffusers",
                            "sort": "downloads",
                            "direction": -1,
                            "limit": limit,
                            "skip": skip,
                        }
                        
                        logger.info(
                            "Fetching HuggingFace models %d-%d...",
                            skip, skip + limit
                        )
                        
                        if progress_callback:
                            progress_callback(skip, self.huggingface_limit)
                        
                        async with session.get(
                            f"{self.HUGGINGFACE_API_BASE}/models",
                            params=params,
                            headers=headers,
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as response:
                            if response.status == 429:
                                # Rate limited (cap at 10 seconds)
                                retry_after = min(int(response.headers.get("Retry-After", "10")), 10)
                                logger.warning("Rate limited, waiting %d seconds (API requested %s)", 
                                             retry_after, response.headers.get("Retry-After", "unknown"))
                                await asyncio.sleep(retry_after)
                                continue
                            
                            if response.status != 200:
                                logger.error("HuggingFace API error: %s", response.status)
                                break
                            
                            models = await response.json()
                        
                        if not models:
                            logger.info("No more models, stopping")
                            break
                        
                        # Process models
                        models_data = []
                        for item in models:
                            try:
                                model_data = self._parse_huggingface_model(item)
                                if model_data:
                                    models_data.append(model_data)
                            except Exception as e:
                                logger.debug("Error parsing model: %s", e)
                                continue
                        
                        if models_data:
                            self._bulk_insert_models(models_data)
                            total_models += len(models_data)
                            logger.info(
                                "Stored %d models (total: %d)",
                                len(models_data), total_models
                            )
                        
                        skip += len(models)
                        
                        # If we got fewer models than requested, we've reached the end
                        if len(models) < limit:
                            logger.info("Reached end of results")
                            break
                        
                        # Rate limiting (2 seconds between requests to avoid 429s)
                        await asyncio.sleep(2.0)
                        
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.error("Error fetching models at offset %d: %s", skip, e)
                        self._sync_error = str(e)
                        skip += limit
                        await asyncio.sleep(2.0)
            
            # Update sync metadata
            self._update_sync_metadata("huggingface", total_models)
            
            logger.info("HuggingFace sync complete: %d models synced", total_models)
            
        except asyncio.CancelledError:
            logger.warning("HuggingFace sync cancelled")
            raise
        except Exception as e:
            logger.error("HuggingFace sync failed: %s", e)
            self._sync_error = str(e)
        finally:
            self._sync_in_progress = False
    
    def _parse_civitai_model(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse CivitAI API response item into model data."""
        try:
            model_id = f"civitai:{item['id']}"
            
            # Get latest version
            versions = item.get("modelVersions", [])
            if not versions:
                return None
            
            latest_version = versions[0]
            
            # Get primary file
            files = latest_version.get("files", [])
            if not files:
                return None
            
            primary_file = files[0]
            
            # Get thumbnail
            images = latest_version.get("images", [])
            thumbnail = images[0].get("url") if images else None
            
            # Get base model
            base_model = latest_version.get("baseModel", "Unknown")
            
            return {
                "id": model_id,
                "source": "civitai",
                "name": item["name"],
                "type": item.get("type", "Unknown"),
                "base_model": base_model,
                "creator": item.get("creator", {}).get("username", "Unknown"),
                "description": item.get("description", "")[:1000],
                "tags": json.dumps(item.get("tags", [])),
                "nsfw": 1 if item.get("nsfw", False) else 0,
                "download_url": primary_file.get("downloadUrl", ""),
                "file_size": int(primary_file.get("sizeKB", 0) * 1024),
                "thumbnail_url": thumbnail,
                "rating": item.get("stats", {}).get("rating", 0.0),
                "download_count": item.get("stats", {}).get("downloadCount", 0),
                "created_at": item.get("createdAt"),
                "updated_at": item.get("updatedAt"),
                "metadata": json.dumps({
                    "civitai_id": item["id"],
                    "version_id": latest_version["id"],
                    "version_name": latest_version.get("name"),
                }),
                "last_synced": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.debug("Error parsing CivitAI model: %s", e)
            return None
    
    def _parse_huggingface_model(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse HuggingFace API response item into model data."""
        try:
            model_id = f"huggingface:{item['id']}"
            
            # Extract author and name
            parts = item["id"].split("/")
            author = parts[0] if len(parts) > 1 else "Unknown"
            name = parts[1] if len(parts) > 1 else item["id"]
            
            # Get tags
            tags = item.get("tags", [])
            
            # Determine base model from tags
            base_model = "Unknown"
            if "stable-diffusion-xl" in tags or "sdxl" in tags:
                base_model = "SDXL"
            elif "stable-diffusion" in tags:
                base_model = "SD 1.5"
            
            return {
                "id": model_id,
                "source": "huggingface",
                "name": name,
                "type": "Checkpoint",
                "base_model": base_model,
                "creator": author,
                "description": "",
                "tags": json.dumps(tags),
                "nsfw": 0,
                "download_url": f"https://huggingface.co/{item['id']}",
                "file_size": 0,
                "thumbnail_url": None,
                "rating": 0.0,
                "download_count": item.get("downloads", 0),
                "created_at": item.get("createdAt"),
                "updated_at": item.get("lastModified"),
                "metadata": json.dumps({
                    "huggingface_id": item["id"],
                    "likes": item.get("likes", 0),
                    "pipeline_tag": item.get("pipeline_tag"),
                }),
                "last_synced": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.debug("Error parsing HuggingFace model: %s", e)
            return None
    
    def _bulk_insert_models(self, models_data: List[Dict[str, Any]]) -> None:
        """Bulk insert or update models in database."""
        conn = sqlite3.connect(str(self.database_path))
        cursor = conn.cursor()
        
        try:
            cursor.executemany("""
                INSERT OR REPLACE INTO models (
                    id, source, name, type, base_model, creator, description,
                    tags, nsfw, download_url, file_size, thumbnail_url,
                    rating, download_count, created_at, updated_at,
                    metadata, last_synced
                ) VALUES (
                    :id, :source, :name, :type, :base_model, :creator, :description,
                    :tags, :nsfw, :download_url, :file_size, :thumbnail_url,
                    :rating, :download_count, :created_at, :updated_at,
                    :metadata, :last_synced
                )
            """, models_data)
            
            conn.commit()
        except Exception as e:
            logger.error("Error bulk inserting models: %s", e)
            conn.rollback()
        finally:
            conn.close()
    
    def _update_sync_metadata(self, source: str, record_count: int) -> None:
        """Update sync metadata table."""
        conn = sqlite3.connect(str(self.database_path))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO sync_metadata (source, last_sync, record_count, error)
                VALUES (?, ?, ?, ?)
            """, (source, datetime.utcnow().isoformat(), record_count, self._sync_error))
            
            conn.commit()
        except Exception as e:
            logger.error("Error updating sync metadata: %s", e)
        finally:
            conn.close()
    
    def search(
        self,
        source: Optional[str] = None,
        query: Optional[str] = None,
        types: Optional[List[str]] = None,
        base_model: Optional[str] = None,
        nsfw: Optional[bool] = None,
        sort: str = "rating",
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Search cached models.
        
        Args:
            source: Filter by source (civitai or huggingface)
            query: Text search query (searches name, description, tags, creator)
            types: Filter by model types
            base_model: Filter by base model
            nsfw: Filter by NSFW status
            sort: Sort field (rating, download_count, name, created_at)
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            Tuple of (models, total_count)
        """
        conn = sqlite3.connect(str(self.database_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            # Build query
            where_clauses = []
            params = []
            
            if source:
                where_clauses.append("m.source = ?")
                params.append(source)
            
            if types:
                placeholders = ",".join("?" * len(types))
                where_clauses.append(f"m.type IN ({placeholders})")
                params.extend(types)
            
            if base_model:
                where_clauses.append("m.base_model = ?")
                params.append(base_model)
            
            # NSFW filter:
            # - nsfw=True: Include ALL models (NSFW + SFW) - no filter needed
            # - nsfw=False: Only show SFW models (nsfw=0)
            # - nsfw=None: Include ALL models - no filter needed
            if nsfw is False:
                where_clauses.append("m.nsfw = 0")
            
            # Full-text search if query provided
            if query and query.strip():
                if self._has_fts5:
                    # Use FTS5 table for fast text search
                    fts_query = f"""
                        SELECT m.* FROM models m
                        INNER JOIN models_fts fts ON m.rowid = fts.rowid
                        WHERE fts MATCH ?
                    """
                    
                    # Add filters
                    if where_clauses:
                        fts_query += " AND " + " AND ".join(where_clauses)
                    
                    # Add sorting
                    fts_query += f" ORDER BY m.{sort} DESC"
                    
                    # Add pagination
                    fts_query += " LIMIT ? OFFSET ?"
                    
                    # Execute
                    search_params = [query] + params + [limit, offset]
                    cursor.execute(fts_query, search_params)
                    models = [dict(row) for row in cursor.fetchall()]
                    
                    # Get total count
                    count_query = f"""
                        SELECT COUNT(*) FROM models m
                        INNER JOIN models_fts fts ON m.rowid = fts.rowid
                        WHERE fts MATCH ?
                    """
                    if where_clauses:
                        count_query += " AND " + " AND ".join(where_clauses)
                    
                    cursor.execute(count_query, [query] + params)
                    total = cursor.fetchone()[0]
                    
                else:
                    # Fallback to LIKE search (slower but works without FTS5)
                    search_term = f"%{query}%"
                    search_where = "(m.name LIKE ? OR m.description LIKE ? OR m.tags LIKE ? OR m.creator LIKE ?)"
                    search_params = [search_term, search_term, search_term, search_term]
                    
                    if where_clauses:
                        where_clauses.insert(0, search_where)
                        params = search_params + params
                    else:
                        where_clauses.append(search_where)
                        params = search_params
                    
                    base_query = "SELECT * FROM models m WHERE " + " AND ".join(where_clauses)
                    base_query += f" ORDER BY {sort} DESC LIMIT ? OFFSET ?"
                    
                    cursor.execute(base_query, params + [limit, offset])
                    models = [dict(row) for row in cursor.fetchall()]
                    
                    # Get total count
                    count_query = "SELECT COUNT(*) FROM models m WHERE " + " AND ".join(where_clauses)
                    cursor.execute(count_query, params)
                    total = cursor.fetchone()[0]
                
            else:
                # Regular query without text search
                base_query = "SELECT * FROM models m"
                
                if where_clauses:
                    base_query += " WHERE " + " AND ".join(where_clauses)
                
                base_query += f" ORDER BY {sort} DESC LIMIT ? OFFSET ?"
                
                cursor.execute(base_query, params + [limit, offset])
                models = [dict(row) for row in cursor.fetchall()]
                
                # Get total count
                count_query = "SELECT COUNT(*) FROM models m"
                if where_clauses:
                    count_query += " WHERE " + " AND ".join(where_clauses)
                
                cursor.execute(count_query, params)
                total = cursor.fetchone()[0]
            
            # Parse JSON fields and transform data
            for model in models:
                model["tags"] = json.loads(model["tags"]) if model["tags"] else []
                model["metadata"] = json.loads(model["metadata"]) if model["metadata"] else {}
                model["nsfw"] = bool(model["nsfw"])
                
                # Strip source prefix from ID for CivitAI models
                # Database stores as "civitai:12345", API needs integer 12345
                if model.get("source") == "civitai" and model.get("id"):
                    model_id_str = str(model["id"])
                    if ":" in model_id_str:
                        model["id"] = int(model_id_str.split(":", 1)[1])
            
            return models, total
            
        except Exception as e:
            logger.error("Search error: %s", e)
            return [], 0
        finally:
            conn.close()
    
    def get_sync_status(self) -> SyncStatus:
        """Get current synchronization status."""
        conn = sqlite3.connect(str(self.database_path))
        cursor = conn.cursor()
        
        try:
            # Get sync metadata
            cursor.execute("SELECT source, last_sync, record_count, error FROM sync_metadata")
            rows = cursor.fetchall()
            
            sync_data = {row[0]: row for row in rows}
            
            civitai_data = sync_data.get("civitai")
            hf_data = sync_data.get("huggingface")
            
            return SyncStatus(
                last_sync_civitai=datetime.fromisoformat(civitai_data[1]) if civitai_data and civitai_data[1] else None,
                last_sync_huggingface=datetime.fromisoformat(hf_data[1]) if hf_data and hf_data[1] else None,
                civitai_models=civitai_data[2] if civitai_data else 0,
                huggingface_models=hf_data[2] if hf_data else 0,
                sync_in_progress=self._sync_in_progress,
                current_page=self._current_page,
                total_pages=self._total_pages,
                error=self._sync_error,
            )
        finally:
            conn.close()
