# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius)

"""
ALICE Model Registry

Handles scanning, registering, and managing Stable Diffusion models.
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelEntry:
    """Information about a registered model."""
    id: str
    name: str
    path: str
    model_type: str  # "sd15", "sdxl", "flux", "custom"
    created: int
    size_mb: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class LoRAEntry:
    """Information about a registered LoRA."""
    id: str
    name: str
    path: str
    created: int
    size_mb: int
    base_model: Optional[str] = None  # Compatible base model type (sd15, sdxl, etc.)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ModelRegistry:
    """
    Registry for managing Stable Diffusion models.
    
    Scans model directories and maintains a registry of available models.
    Models can be .safetensors files or diffusers directories.
    """
    
    MODEL_PREFIX = "sd"  # All SD models use sd/ prefix
    
    def __init__(self, models_dir: Path):
        """
        Initialize the model registry.
        
        Args:
            models_dir: Directory containing model files
        """
        self.models_dir = Path(models_dir)
        self.loras_dir = self.models_dir / "loras"
        self.registry_file = self.models_dir / ".registry.json"
        self.models: Dict[str, ModelEntry] = {}
        self.loras: Dict[str, LoRAEntry] = {}
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loras_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing registry or scan
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from file or perform initial scan."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file) as f:
                    data = json.load(f)
                
                for model_data in data.get("models", []):
                    entry = ModelEntry(
                        id=model_data["id"],
                        name=model_data["name"],
                        path=model_data["path"],
                        model_type=model_data["model_type"],
                        created=model_data["created"],
                        size_mb=model_data["size_mb"]
                    )
                    self.models[entry.id] = entry
                
                for lora_data in data.get("loras", []):
                    entry = LoRAEntry(
                        id=lora_data["id"],
                        name=lora_data["name"],
                        path=lora_data["path"],
                        created=lora_data["created"],
                        size_mb=lora_data["size_mb"],
                        base_model=lora_data.get("base_model")
                    )
                    self.loras[entry.id] = entry
                
                logger.info("Loaded %d models and %d LoRAs from registry", len(self.models), len(self.loras))
            except Exception as e:
                logger.warning("Failed to load registry: %s. Rescanning.", e)
                self.scan_models()
        else:
            logger.info("No registry found. Scanning for models...")
            self.scan_models()
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        try:
            data = {
                "version": "1.1",
                "updated": int(datetime.now().timestamp()),
                "models": [model.to_dict() for model in self.models.values()],
                "loras": [lora.to_dict() for lora in self.loras.values()]
            }
            
            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved registry with %d models and %d LoRAs", len(self.models), len(self.loras))
        except Exception as e:
            logger.error("Failed to save registry: %s", e)
    
    def _detect_model_type(self, path: Path) -> str:
        """
        Detect model type from path.
        
        Args:
            path: Path to model file or directory
            
        Returns:
            Model type string (sd15, sdxl, flux, custom)
        """
        name_lower = path.name.lower()
        
        # Check for common model type indicators
        if "qwen" in name_lower:
            return "qwen"
        elif "xl" in name_lower or "sdxl" in name_lower:
            return "sdxl"
        elif "flux" in name_lower:
            return "flux"
        elif "sd3" in name_lower or "stable-diffusion-3" in name_lower:
            return "sd3"
        elif "1-5" in name_lower or "v1-5" in name_lower or "1.5" in name_lower:
            return "sd15"
        elif "2-1" in name_lower or "v2-1" in name_lower or "2.1" in name_lower:
            return "sd21"
        
        # Check model_index.json for diffusers models
        if path.is_dir():
            model_index = path / "model_index.json"
            if model_index.exists():
                try:
                    with open(model_index) as f:
                        data = json.load(f)
                    
                    class_name = data.get("_class_name", "").lower()
                    if "qwen" in class_name:
                        return "qwen"
                    elif "xl" in class_name:
                        return "sdxl"
                    elif "flux" in class_name:
                        return "flux"
                except Exception:
                    pass
        
        # Default to sd15
        return "sd15"
    
    def _calculate_size(self, path: Path) -> int:
        """
        Calculate total size of model in MB.
        
        Args:
            path: Path to model file or directory
            
        Returns:
            Size in megabytes
        """
        if path.is_file():
            return int(path.stat().st_size / (1024 * 1024))
        
        total = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total += file_path.stat().st_size
        
        return int(total / (1024 * 1024))
    
    def scan_models(self) -> List[ModelEntry]:
        """
        Scan models directory and update registry.
        
        Returns:
            List of discovered models
        """
        logger.info("Scanning for models in: %s", self.models_dir)
        found_models: List[ModelEntry] = []
        
        # Clear existing entries before rescanning to remove stale models
        self.models.clear()
        self.loras.clear()
        
        if not self.models_dir.exists():
            logger.warning("Models directory does not exist: %s", self.models_dir)
            return found_models
        
        # Scan for .safetensors files
        for safetensors_file in self.models_dir.rglob("*.safetensors"):
            if safetensors_file.is_file():
                # Skip files in the loras subdirectory
                if self.loras_dir in safetensors_file.parents or safetensors_file.parent == self.loras_dir:
                    continue
                
                # Skip files that are components inside diffusers model directories
                # A diffusers model has model_index.json at its root - any safetensors files
                # in its subdirectories (text_encoder/, transformer/, vae/ etc.) are components
                is_component = False
                for parent in safetensors_file.parents:
                    if parent == self.models_dir:
                        break  # Don't look above models_dir
                    if (parent / "model_index.json").exists():
                        is_component = True
                        break
                if is_component:
                    continue
                
                # Determine model name: use parent directory name if file is in a subdirectory
                # Otherwise use the filename stem
                if safetensors_file.parent != self.models_dir:
                    model_name = safetensors_file.parent.name
                else:
                    model_name = safetensors_file.stem
                
                model_id = f"{self.MODEL_PREFIX}/{model_name}"
                
                entry = ModelEntry(
                    id=model_id,
                    name=model_name,
                    path=str(safetensors_file),
                    model_type=self._detect_model_type(safetensors_file.parent if safetensors_file.parent != self.models_dir else safetensors_file),
                    created=int(safetensors_file.stat().st_ctime),
                    size_mb=self._calculate_size(safetensors_file)
                )
                
                found_models.append(entry)
                self.models[model_id] = entry
                logger.debug("Found safetensors model: %s", model_id)
        
        # Scan for diffusers directories (contain model_index.json)
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                # Skip the loras directory
                if model_dir == self.loras_dir or model_dir.name == "loras":
                    continue
                
                model_index = model_dir / "model_index.json"
                if model_index.exists():
                    model_id = f"{self.MODEL_PREFIX}/{model_dir.name}"
                    
                    # Skip if already found as safetensors
                    if model_id in self.models:
                        continue
                    
                    entry = ModelEntry(
                        id=model_id,
                        name=model_dir.name,
                        path=str(model_dir),
                        model_type=self._detect_model_type(model_dir),
                        created=int(model_dir.stat().st_ctime),
                        size_mb=self._calculate_size(model_dir)
                    )
                    
                    found_models.append(entry)
                    self.models[model_id] = entry
                    logger.debug("Found diffusers model: %s", model_id)
        
        # Scan for LoRAs
        self._scan_loras()
        
        # Save updated registry
        self._save_registry()
        
        logger.info("Found %d models and %d LoRAs", len(found_models), len(self.loras))
        return found_models
    
    def get_model(self, model_id: str) -> Optional[ModelEntry]:
        """
        Get model entry by ID.
        
        Args:
            model_id: Model identifier (e.g., sd/stable-diffusion-v1-5)
            
        Returns:
            ModelEntry or None if not found
        """
        return self.models.get(model_id)
    
    def get_model_path(self, model_id: str) -> Optional[Path]:
        """
        Get path to model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Path to model or None if not found
        """
        entry = self.get_model(model_id)
        if entry:
            return Path(entry.path)
        return None
    
    def list_models(self) -> List[ModelEntry]:
        """
        List all registered models.
        
        Returns:
            List of all model entries
        """
        return list(self.models.values())
    
    def _scan_loras(self) -> List[LoRAEntry]:
        """
        Scan loras directory for LoRA files.
        
        Returns:
            List of discovered LoRAs
        """
        found_loras: List[LoRAEntry] = []
        
        if not self.loras_dir.exists():
            logger.debug("LoRAs directory does not exist: %s", self.loras_dir)
            return found_loras
        
        # Scan for .safetensors LoRA files
        for lora_file in self.loras_dir.glob("*.safetensors"):
            if lora_file.is_file():
                lora_name = lora_file.stem
                lora_id = f"lora/{lora_name}"
                
                # Detect base model type from filename
                base_model = self._detect_lora_base_model(lora_file)
                
                entry = LoRAEntry(
                    id=lora_id,
                    name=lora_name,
                    path=str(lora_file),
                    created=int(lora_file.stat().st_ctime),
                    size_mb=self._calculate_size(lora_file),
                    base_model=base_model
                )
                
                found_loras.append(entry)
                self.loras[lora_id] = entry
                logger.debug("Found LoRA: %s", lora_id)
        
        logger.info("Found %d LoRAs", len(found_loras))
        return found_loras
    
    def _detect_lora_base_model(self, path: Path) -> Optional[str]:
        """
        Detect compatible base model type for LoRA.
        
        Args:
            path: Path to LoRA file
            
        Returns:
            Base model type or None
        """
        name_lower = path.name.lower()
        
        if "xl" in name_lower or "sdxl" in name_lower:
            return "sdxl"
        elif "flux" in name_lower:
            return "flux"
        elif "sd3" in name_lower:
            return "sd3"
        elif "sd15" in name_lower or "sd1" in name_lower or "1.5" in name_lower:
            return "sd15"
        
        # Default to None (compatible with any)
        return None
    
    def get_lora(self, lora_id: str) -> Optional[LoRAEntry]:
        """
        Get LoRA entry by ID.
        
        Args:
            lora_id: LoRA identifier (e.g., lora/style-lora)
            
        Returns:
            LoRAEntry or None if not found
        """
        return self.loras.get(lora_id)
    
    def get_lora_path(self, lora_id: str) -> Optional[Path]:
        """
        Get path to LoRA by ID or fuzzy name match.
        
        Supports:
        - Exact ID match (e.g., "lora/my_lora_v1.0")
        - ID without prefix (e.g., "my_lora_v1.0")  
        - Partial name match (e.g., "add_detail" matches "Add_More_Details_-_Detail_Enhancer...")
        - Case-insensitive matching
        
        Args:
            lora_id: LoRA identifier or partial name
            
        Returns:
            Path to LoRA or None if not found
        """
        # Try exact match first
        entry = self.get_lora(lora_id)
        if entry:
            return Path(entry.path)
        
        # Try with lora/ prefix
        if not lora_id.startswith("lora/"):
            entry = self.get_lora(f"lora/{lora_id}")
            if entry:
                return Path(entry.path)
        
        # Try fuzzy matching - search for LoRAs containing the search term
        search_lower = lora_id.lower().replace("_", "").replace("-", "").replace(" ", "")
        for lora_entry in self.loras.values():
            lora_name_lower = lora_entry.name.lower().replace("_", "").replace("-", "").replace(" ", "")
            # Check if search term is contained in LoRA name
            if search_lower in lora_name_lower:
                logger.info("Fuzzy matched LoRA '%s' -> '%s'", lora_id, lora_entry.name)
                return Path(lora_entry.path)
        
        return None
    
    def list_loras(self) -> List[LoRAEntry]:
        """
        List all registered LoRAs.
        
        Returns:
            List of all LoRA entries
        """
        return list(self.loras.values())
    
    def refresh(self) -> List[ModelEntry]:
        """
        Refresh model registry by rescanning.
        
        Returns:
            List of all models after refresh
        """
        self.models.clear()
        self.loras.clear()
        return self.scan_models()
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model from disk and registry.
        
        Args:
            model_id: Model identifier (e.g., sd/model-name)
            
        Returns:
            True if deleted, False if not found
        """
        entry = self.models.get(model_id)
        if not entry:
            logger.warning("Model not found for deletion: %s", model_id)
            return False
        
        model_path = Path(entry.path)
        
        try:
            if model_path.is_file():
                # Delete safetensors file
                model_path.unlink()
                logger.info("Deleted model file: %s", model_path)
                
                # If the model was in a subdirectory, check if we should delete the directory
                if model_path.parent != self.models_dir:
                    parent = model_path.parent
                    # Only delete if empty or only contains our model
                    remaining = list(parent.iterdir())
                    if not remaining or (len(remaining) == 1 and remaining[0].name.startswith(".")):
                        import shutil
                        shutil.rmtree(parent)
                        logger.info("Deleted empty model directory: %s", parent)
            elif model_path.is_dir():
                # Delete diffusers directory
                import shutil
                shutil.rmtree(model_path)
                logger.info("Deleted model directory: %s", model_path)
            else:
                logger.warning("Model path does not exist: %s", model_path)
            
            # Remove from registry
            del self.models[model_id]
            self._save_registry()
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete model %s: %s", model_id, e)
            return False
    
    def delete_lora(self, lora_id: str) -> bool:
        """
        Delete a LoRA from disk and registry.
        
        Args:
            lora_id: LoRA identifier (e.g., lora/style-lora)
            
        Returns:
            True if deleted, False if not found
        """
        entry = self.loras.get(lora_id)
        if not entry:
            logger.warning("LoRA not found for deletion: %s", lora_id)
            return False
        
        lora_path = Path(entry.path)
        
        try:
            if lora_path.is_file():
                lora_path.unlink()
                logger.info("Deleted LoRA file: %s", lora_path)
            else:
                logger.warning("LoRA path does not exist: %s", lora_path)
            
            # Remove from registry
            del self.loras[lora_id]
            self._save_registry()
            
            return True
            
        except Exception as e:
            logger.error("Failed to delete LoRA %s: %s", lora_id, e)
            return False
