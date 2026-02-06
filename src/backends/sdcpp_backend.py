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
        enable_vae_tiling: bool = False,
        enable_vae_slicing: bool = False,
        vae_decode_cpu: bool = False,
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
            enable_vae_tiling: Enable VAE tiling for memory optimization
            enable_vae_slicing: Enable VAE slicing (Note: sd.cpp uses tiling instead)
            vae_decode_cpu: Keep VAE on CPU (for low VRAM systems)
        """
        self.images_dir = Path(images_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        self.default_steps = default_steps
        self.default_guidance_scale = default_guidance_scale
        self.default_scheduler = default_scheduler
        self.default_width = default_width
        self.default_height = default_height
        self.sdcpp_threads = sdcpp_threads
        
        # VAE optimizations
        self.enable_vae_tiling = enable_vae_tiling
        self.enable_vae_slicing = enable_vae_slicing  # sd.cpp uses tiling, map to --vae-tiling
        self.vae_decode_cpu = vae_decode_cpu
        
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
        
        # Validate and resolve model path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Handle different model formats
        is_diffusers_dir = False
        diffusers_components = {}
        
        if model_path.is_dir():
            # Check for single-file format (model.safetensors in root)
            safetensors_path = model_path / "model.safetensors"
            if safetensors_path.exists():
                logger.debug(f"Using model.safetensors from directory: {model_path}")
                model_path = safetensors_path
            else:
                # Check for diffusers format (separate component directories)
                is_diffusers_dir = True
                logger.debug(f"Detected diffusers directory format: {model_path}")
                
                # Map diffusers components to sd.cpp paths
                unet_path = model_path / "unet" / "diffusion_pytorch_model.safetensors"
                vae_path = model_path / "vae" / "diffusion_pytorch_model.safetensors"
                
                if not unet_path.exists():
                    raise FileNotFoundError(
                        f"Diffusers directory missing required unet model: {unet_path}"
                    )
                
                if not vae_path.exists():
                    raise FileNotFoundError(
                        f"Diffusers directory missing required VAE model: {vae_path}"
                    )
                
                diffusers_components["diffusion_model"] = unet_path
                diffusers_components["vae"] = vae_path
                
                # Check for text encoders (SDXL has clip_l and clip_g, SD1.5 has just one)
                text_encoder_path = model_path / "text_encoder" / "model.safetensors"
                text_encoder_2_path = model_path / "text_encoder_2" / "model.safetensors"
                
                if text_encoder_path.exists():
                    diffusers_components["clip_l"] = text_encoder_path
                
                if text_encoder_2_path.exists():
                    diffusers_components["clip_g"] = text_encoder_2_path
                    logger.debug("Detected SDXL model (dual text encoders)")
                else:
                    logger.debug("Detected SD1.5 model (single text encoder)")
        
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
        
        # Build command based on model format
        cmd = [str(self.sdcpp_binary)]
        
        if is_diffusers_dir:
            # For diffusers format, we need BOTH -m (directory) and component paths
            # The -m tells sd.cpp it's diffusers format, components override defaults
            cmd.extend(["-m", str(model_path)])
            cmd.extend(["--diffusion-model", str(diffusers_components["diffusion_model"])])
            cmd.extend(["--vae", str(diffusers_components["vae"])])
            
            if "clip_l" in diffusers_components:
                cmd.extend(["--clip_l", str(diffusers_components["clip_l"])])
            
            if "clip_g" in diffusers_components:
                cmd.extend(["--clip_g", str(diffusers_components["clip_g"])])
            
            logger.info(f"Loading diffusers model from: {model_path.name}")
        else:
            # Use single model file
            cmd.extend(["-m", str(model_path)])
            logger.info(f"Loading single-file model: {model_path.name}")
        
        # Add generation parameters
        cmd.extend([
            "-p", prompt,
            "-n", negative_prompt or "",
            "--steps", str(steps),
            "--cfg-scale", str(guidance_scale),
            "-W", str(width),
            "-H", str(height),
            "--sampling-method", sdcpp_scheduler,
            "-o", str(output_path),
            "-t", str(self.sdcpp_threads),
        ])
        
        # VAE optimizations
        if self.enable_vae_tiling or self.enable_vae_slicing:
            cmd.append("--vae-tiling")
            logger.debug("Enabled VAE tiling for memory optimization")
        
        if self.vae_decode_cpu:
            cmd.append("--vae-on-cpu")
            logger.debug("VAE will run on CPU (for low VRAM)")
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # LoRA support check
        if lora_paths:
            logger.warning("LoRA support not yet implemented for sd.cpp backend")
        
        # Multi-image warning
        if num_images > 1:
            logger.warning(f"Requested {num_images} images, but sd.cpp backend only supports 1 per call")
        
        # Log command (sanitized)
        logger.info(f"Executing sd-cli: steps={steps}, scheduler={sdcpp_scheduler}, size={width}x{height}")
        logger.debug(f"Full command: {' '.join(cmd)}")
        
        # Execute subprocess with cancellation support
        loop = asyncio.get_event_loop()
        process = None
        
        async def _run_subprocess_with_cancellation():
            """Run sd-cli with periodic cancellation checks."""
            nonlocal process
            
            # Start subprocess - redirect stderr to avoid buffer filling (sd.cpp is very verbose)
            # We keep stdout for progress info but let stderr go to console/logs
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=None,  # Let stderr go to console - avoids deadlock from buffer filling
            )
            
            # Poll process while checking for cancellation
            try:
                while True:
                    # Check if process finished
                    returncode = process.poll()
                    if returncode is not None:
                        # Process completed - read stdout
                        stdout, _ = process.communicate()
                        return subprocess.CompletedProcess(
                            args=cmd,
                            returncode=returncode,
                            stdout=stdout,
                            stderr=b"",  # stderr was not captured
                        )
                    
                    # Check cancellation token
                    if cancellation_token and cancellation_token.is_cancelled():
                        logger.info("Cancellation requested, terminating sd-cli process")
                        process.terminate()
                        
                        # Wait up to 5 seconds for graceful termination
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning("sd-cli did not terminate gracefully, killing process")
                            process.kill()
                            process.wait()
                        
                        # Clean up output file if created
                        if output_path.exists():
                            output_path.unlink()
                        raise CancellationError("Generation cancelled")
                    
                    # Sleep briefly before next check (non-blocking)
                    await asyncio.sleep(0.5)
            
            except Exception as e:
                # Ensure process is cleaned up on any error
                if process and process.poll() is None:
                    process.kill()
                    process.wait()
                raise
        
        try:
            # Run with timeout
            result = await asyncio.wait_for(
                _run_subprocess_with_cancellation(),
                timeout=30 * 60  # 30 minute timeout
            )
            
            # Check exit code
            if result.returncode != 0:
                stdout_msg = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
                logger.error(f"sd-cli failed with code {result.returncode}")
                logger.error(f"sd-cli stdout: {stdout_msg[:500]}")  # First 500 chars
                raise RuntimeError(f"Image generation failed (exit code {result.returncode}). Check logs for details.")
            
            # Verify output file exists
            if not output_path.exists():
                stdout_msg = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
                logger.error(f"sd-cli succeeded but output file not found: {output_path}")
                logger.error(f"sd-cli stdout: {stdout_msg[:1000]}")
                raise RuntimeError(f"Image generation completed but output file not created. Check logs for details.")
            
            elapsed = time.time() - start_time
            
            # Parse stdout for any useful info
            stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
            
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
        
        except asyncio.TimeoutError:
            logger.error(f"sd-cli timed out after 30 minutes")
            # Ensure process is killed
            if process and process.poll() is None:
                process.kill()
                process.wait()
            # Clean up partial output
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError("Image generation timed out (30 minutes)")
        
        except Exception as e:
            logger.error(f"sd-cli execution failed: {e}")
            # Ensure process is killed
            if process and process.poll() is None:
                process.kill()
                process.wait()
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
        # Default response
        info = {
            "type": "vulkan",
            "name": "Unknown Vulkan Device",
            "driver": "Vulkan",
            "backend": "stable-diffusion.cpp",
            "gpu_available": True,
            "stats_available": False,
            "utilization": 0.0,
            "memory_used": "N/A",
            "memory_total": "N/A",
        }
        
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
                for line in output.split('\n'):
                    if 'deviceName' in line or 'GPU' in line:
                        parts = line.split('=')
                        if len(parts) > 1:
                            info["name"] = parts[1].strip()
                            break
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # For AMD GPUs, read actual GPU stats from sysfs
        # This works regardless of backend (Vulkan, PyTorch, etc.)
        if self._is_amd_gpu():
            try:
                # Read memory usage
                allocated = self._read_amd_memory_usage()
                if allocated > 0:
                    info["memory_used"] = f"{allocated / (1024**3):.1f} GB"
                    info["stats_available"] = True
                
                # Read total memory from sysfs
                total = self._read_amd_memory_total()
                if total > 0:
                    info["memory_total"] = f"{total / (1024**3):.1f} GB"
                
                # Read GPU utilization
                gpu_busy = self._read_amd_gpu_busy()
                if gpu_busy >= 0:
                    info["utilization"] = gpu_busy / 100.0
                    info["stats_available"] = True
                    
            except Exception as e:
                logger.debug("Failed to read AMD GPU stats from sysfs: %s", e)
        
        return info
    
    def _is_amd_gpu(self) -> bool:
        """Check if current GPU is AMD/ATI."""
        try:
            return Path("/sys/class/drm/card0/device/vendor").exists()
        except Exception:
            return False
    
    def _read_amd_memory_usage(self) -> int:
        """
        Read AMD GPU memory usage from sysfs.
        
        For AMD APUs, shared system memory is tracked via GTT.
        For discrete GPUs, use VRAM.
        
        Returns:
            Memory usage in bytes, or 0 if reading fails
        """
        try:
            # Try GTT (Graphics Translation Table) first - used by APUs
            gtt_used_path = Path("/sys/class/drm/card0/device/mem_info_gtt_used")
            if gtt_used_path.exists():
                return int(gtt_used_path.read_text().strip())
            
            # Fall back to VRAM for discrete GPUs
            vram_used_path = Path("/sys/class/drm/card0/device/mem_info_vram_used")
            if vram_used_path.exists():
                return int(vram_used_path.read_text().strip())
                
        except Exception as e:
            logger.debug("Failed to read AMD memory usage from sysfs: %s", e)
        
        return 0
    
    def _read_amd_memory_total(self) -> int:
        """
        Read AMD GPU total memory from sysfs.
        
        Returns:
            Total memory in bytes, or 0 if reading fails
        """
        try:
            # Try GTT total first (for APUs)
            gtt_total_path = Path("/sys/class/drm/card0/device/mem_info_gtt_total")
            if gtt_total_path.exists():
                return int(gtt_total_path.read_text().strip())
            
            # Fall back to VRAM total for discrete GPUs
            vram_total_path = Path("/sys/class/drm/card0/device/mem_info_vram_total")
            if vram_total_path.exists():
                return int(vram_total_path.read_text().strip())
                
        except Exception as e:
            logger.debug("Failed to read AMD total memory from sysfs: %s", e)
        
        return 0
    
    def _read_amd_gpu_busy(self) -> int:
        """
        Read AMD GPU busy percentage from sysfs.
        
        Returns:
            GPU busy percentage (0-100), or -1 if reading fails
        """
        try:
            busy_path = Path("/sys/class/drm/card0/device/gpu_busy_percent")
            if busy_path.exists():
                return int(busy_path.read_text().strip())
        except Exception as e:
            logger.debug("Failed to read AMD GPU busy from sysfs: %s", e)
        
        return -1
    
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
