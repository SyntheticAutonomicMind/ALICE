# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
Vulkan Backend for ALICE using stable-diffusion.cpp

Provides universal AMD GPU support via Vulkan, bypassing ROCm entirely.
Executes sd-cli binary as subprocess for image generation.
"""

import asyncio
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

from .base import BaseBackend
from ..cancellation import CancellationToken, CancellationError

logger = logging.getLogger(__name__)


# Map ALICE scheduler names to sd.cpp equivalents
SCHEDULER_MAPPING = {
    "dpm++": "dpm++2m",
    "dpm++_karras": "dpm++2m",
    "dpm++_sde": "dpm++2s_a",
    "dpm++_sde_karras": "dpm++2s_a",
    "dpm++2m": "dpm++2m",
    "dpm++2mv2": "dpm++2mv2",
    "dpm++2s_a": "dpm++2s_a",
    "euler": "euler",
    "euler_a": "euler_a",
    "euler_ancestral": "euler_a",
    "heun": "heun",
    "dpm2": "dpm2",
    "ddim": "ddim_trailing",
    "lcm": "lcm",
    "pndm": "ipndm",
    "ipndm": "ipndm",
}


class SDCppBackend(BaseBackend):
    """
    Vulkan backend using stable-diffusion.cpp.
    
    Executes sd-cli binary as subprocess for image generation.
    Works on ANY AMD GPU (universal Vulkan support).
    """
    
    def __init__(
        self,
        images_dir: Path,
        default_steps: int = 20,
        default_guidance_scale: float = 7.5,
        default_scheduler: str = "euler_a",
        default_width: int = 512,
        default_height: int = 512,
        sdcpp_binary: Optional[Path] = None,
        sdcpp_threads: int = 8,
        vae_on_cpu: bool = False,
        vae_decode_cpu: bool = False,  # Alias for vae_on_cpu (from PyTorch config)
        **kwargs
    ):
        """
        Initialize Vulkan backend.
        
        Args:
            images_dir: Directory to save generated images
            default_steps: Default number of inference steps
            default_guidance_scale: Default CFG scale
            default_scheduler: Default sampling method
            default_width: Default image width
            default_height: Default image height
            sdcpp_binary: Path to sd-cli binary (auto-detected if None)
            sdcpp_threads: CPU threads for Vulkan operations
            vae_on_cpu: Keep VAE in CPU to reduce VRAM usage
            vae_decode_cpu: Alias for vae_on_cpu (PyTorch config compatibility)
        """
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_steps = default_steps
        self.default_guidance_scale = default_guidance_scale
        self.default_scheduler = default_scheduler
        self.default_width = default_width
        self.default_height = default_height
        self.sdcpp_threads = sdcpp_threads
        self.vae_on_cpu = vae_on_cpu or vae_decode_cpu  # Use either parameter
        
        # Log initialization parameters
        logger.info(f"SDCppBackend init: vae_on_cpu={vae_on_cpu}, vae_decode_cpu={vae_decode_cpu}, final={self.vae_on_cpu}")
        
        # Auto-detect or validate binary
        if sdcpp_binary is None:
            self.sdcpp_binary = self._find_sdcpp_binary()
        else:
            self.sdcpp_binary = Path(sdcpp_binary)
            if not self.sdcpp_binary.exists():
                raise FileNotFoundError(f"sd-cli binary not found at {self.sdcpp_binary}")
            if not self.sdcpp_binary.is_file():
                raise ValueError(f"sd-cli path is not a file: {self.sdcpp_binary}")
        
        # sd.cpp doesn't pre-load models, but we track the "current" one
        self._current_model: Optional[str] = None
        
        logger.info(f"Initialized SDCppBackend with binary: {self.sdcpp_binary}")
        logger.info(f"Vulkan threads: {self.sdcpp_threads}")
    
    @staticmethod
    def _find_sdcpp_binary() -> Path:
        """
        Auto-detect sd-cli binary location.
        
        Searches in priority order:
        1. /usr/local/bin/sd-cli (system installation)
        2. /usr/bin/sd-cli (package manager)
        3. ~/sd.cpp/build/bin/sd-cli (user build)
        4. /home/deck/sd.cpp/build/bin/sd-cli (Steam Deck specific)
        
        Returns:
            Path to sd-cli binary
            
        Raises:
            RuntimeError: If binary not found in any location
        """
        search_paths = [
            Path("/usr/local/bin/sd-cli"),
            Path("/usr/bin/sd-cli"),
            Path.home() / "sd.cpp" / "build" / "bin" / "sd-cli",
            Path("/home/deck/sd.cpp/build/bin/sd-cli"),
        ]
        
        for path in search_paths:
            if path.exists() and path.is_file():
                logger.info(f"Found sd-cli binary at {path}")
                return path
        
        raise RuntimeError(
            "sd-cli binary not found. Install stable-diffusion.cpp and build it, "
            "or use scripts/build_sdcpp.sh to build automatically. "
            f"Searched: {[str(p) for p in search_paths]}"
        )
    
    async def load_model(self, model_path: Path) -> None:
        """
        Load model (no-op for sd.cpp - models loaded per-request).
        
        Just validates the model file exists and stores the path.
        
        Args:
            model_path: Path to model file (.safetensors or directory)
            
        Raises:
            FileNotFoundError: If model path doesn't exist
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self._current_model = str(model_path)
        logger.info(f"Model path set: {model_path.name}")
    
    async def unload_model(self) -> None:
        """
        Unload model (no-op for sd.cpp).
        
        sd.cpp doesn't keep models in memory between requests.
        """
        self._current_model = None
        logger.debug("Model path cleared")
    
    async def generate_image(
        self,
        model_path: Path,
        prompt: str,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        scheduler: Optional[str] = None,
        num_images: int = 1,
        lora_paths: Optional[List[Path]] = None,
        lora_scales: Optional[List[float]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Tuple[List[Path], Dict[str, Any]]:
        """
        Generate image(s) using sd-cli subprocess.
        
        Args:
            model_path: Path to model file
            prompt: Positive prompt text
            negative_prompt: Negative prompt text
            steps: Number of inference steps
            guidance_scale: CFG scale
            width: Image width in pixels
            height: Image height in pixels
            seed: Random seed
            scheduler: Sampling method name
            num_images: Number of images to generate (currently limited to 1)
            lora_paths: List of LoRA model paths (not yet supported)
            lora_scales: List of LoRA scales (not yet supported)
            cancellation_token: Token for cancelling generation
            
        Returns:
            Tuple of (image_paths, metadata_dict)
            
        Raises:
            FileNotFoundError: If model doesn't exist
            RuntimeError: If sd-cli execution fails
            CancellationError: If generation cancelled
        """
        start_time = time.time()
        
        # Log received parameters
        logger.info(f"generate_image called: width={width}, height={height}, steps={steps}, guidance={guidance_scale}")
        
        # Validate model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Apply defaults
        steps = steps or self.default_steps
        guidance_scale = guidance_scale or self.default_guidance_scale
        width = width or self.default_width
        height = height or self.default_height
        scheduler = scheduler or self.default_scheduler
        
        # Map scheduler name
        sdcpp_scheduler = SCHEDULER_MAPPING.get(scheduler, "euler_a")
        if scheduler not in SCHEDULER_MAPPING:
            logger.warning(f"Unknown scheduler '{scheduler}', using 'euler_a'")
        
        # Generate unique output filename
        image_id = uuid.uuid4().hex[:12]
        output_path = self.images_dir / f"{image_id}.png"
        
        # Build command
        cmd = [
            str(self.sdcpp_binary),
            "-m", str(model_path),
            "-p", prompt,
            "-n", negative_prompt or "",
            "--steps", str(steps),
            "--cfg-scale", str(guidance_scale),
            "-W", str(width),
            "-H", str(height),
            "--sampling-method", sdcpp_scheduler,
            "-o", str(output_path),
            "-t", str(self.sdcpp_threads),
        ]
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # VAE on CPU for low VRAM systems
        if self.vae_on_cpu:
            cmd.append("--vae-on-cpu")
            logger.info("VAE on CPU enabled (vae_on_cpu=%s)", self.vae_on_cpu)
        else:
            logger.warning("VAE on CPU NOT enabled (vae_on_cpu=%s)", self.vae_on_cpu)
        
        # LoRA support check
        if lora_paths:
            logger.warning("LoRA support not yet implemented for sd.cpp backend")
        
        # Multi-image warning
        if num_images > 1:
            logger.warning(f"Requested {num_images} images, but sd.cpp backend only supports 1 per call")
        
        # Log command (sanitized)
        logger.info(f"Executing sd-cli: steps={steps}, scheduler={sdcpp_scheduler}, size={width}x{height}")
        logger.debug(f"Full command: {' '.join(cmd)}")
        
        # Execute subprocess in thread pool (don't block event loop)
        loop = asyncio.get_event_loop()
        
        def _run_subprocess():
            """Run sd-cli and capture output."""
            return subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30 * 60,  # 30 minute timeout
                check=False  # Don't raise on non-zero exit
            )
        
        try:
            # Run in executor
            result = await loop.run_in_executor(None, _run_subprocess)
            
            # Check cancellation
            if cancellation_token and cancellation_token.is_cancelled():
                # Clean up output file if created
                if output_path.exists():
                    output_path.unlink()
                raise CancellationError("Generation cancelled")
            
            # Check exit code
            if result.returncode != 0:
                error_msg = result.stderr.decode('utf-8', errors='replace')
                logger.error(f"sd-cli failed with code {result.returncode}: {error_msg}")
                raise RuntimeError(f"Image generation failed: {error_msg}")
            
            # Verify output file exists
            if not output_path.exists():
                logger.error(f"sd-cli succeeded but output file not found: {output_path}")
                raise RuntimeError("Image generation completed but output file not created")
            
            elapsed = time.time() - start_time
            
            # Parse stdout for any useful info
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            
            # Build metadata
            metadata = {
                "backend": "vulkan",
                "backend_name": self.get_backend_name(),
                "model": model_path.name,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "seed": seed,
                "scheduler": scheduler,
                "sdcpp_scheduler": sdcpp_scheduler,
                "generation_time": round(elapsed, 2),
                "time_per_step": round(elapsed / steps, 2) if steps > 0 else 0,
            }
            
            logger.info(f"Image generated successfully in {elapsed:.2f}s ({elapsed/steps:.2f}s/step)")
            logger.debug(f"Output: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
            
            # Update current model
            self._current_model = str(model_path)
            
            return ([output_path], metadata)
        
        except subprocess.TimeoutExpired:
            logger.error(f"sd-cli timed out after 30 minutes")
            raise RuntimeError("Image generation timed out (30 minutes)")
        
        except Exception as e:
            logger.error(f"sd-cli execution failed: {e}")
            # Clean up partial output
            if output_path.exists():
                output_path.unlink()
            raise
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get Vulkan GPU information.
        
        Returns:
            Dict with GPU details (name, type, driver, VRAM if available, gpu_available)
        """
        try:
            # Try to get Vulkan device info via vulkaninfo
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                check=False
            )
            
            if result.returncode == 0:
                output = result.stdout.decode('utf-8', errors='replace')
                
                # Parse device name (basic parsing)
                device_name = "Unknown Vulkan Device"
                for line in output.split('\n'):
                    if 'deviceName' in line or 'GPU' in line:
                        parts = line.split('=')
                        if len(parts) > 1:
                            device_name = parts[1].strip()
                            break
                
                return {
                    "type": "vulkan",
                    "name": device_name,
                    "driver": "Vulkan",
                    "backend": "stable-diffusion.cpp",
                    "gpu_available": True,
                    "stats_available": False,
                }
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("vulkaninfo not available, using generic Vulkan info")
        
        # Fallback
        return {
            "type": "vulkan",
            "name": "Vulkan Device",
            "driver": "Vulkan",
            "backend": "stable-diffusion.cpp",
            "gpu_available": True,
            "stats_available": False,
        }
    
    @property
    def current_model(self) -> Optional[str]:
        """Get currently loaded model path (or last used model)."""
        return self._current_model
    
    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded (always False for sd.cpp)."""
        # sd.cpp doesn't keep models in memory, so technically never "loaded"
        # But we return True if we have a current model path set
        return self._current_model is not None
    
    @staticmethod
    def is_available() -> bool:
        """
        Check if sd.cpp backend is available.
        
        Returns:
            True if sd-cli binary exists and is executable
        """
        try:
            SDCppBackend._find_sdcpp_binary()
            return True
        except RuntimeError:
            return False
    
    @staticmethod
    def get_backend_name() -> str:
        """
        Get human-readable backend name.
        
        Returns:
            Backend name string
        """
        return "Vulkan (stable-diffusion.cpp)"
