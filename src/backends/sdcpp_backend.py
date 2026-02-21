# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
Vulkan Backend for ALICE using stable-diffusion.cpp

Provides universal AMD GPU support via Vulkan, bypassing ROCm entirely.
Executes sd-cli binary as subprocess for image generation.
"""

import asyncio
import glob
import json
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
        enable_vae_tiling: bool = False,  # WARNING: VAE tiling is SLOWER on Vulkan for typical resolutions (<2048px)
        enable_mmap: bool = False,
        keep_clip_on_cpu: bool = False,
        enable_model_cpu_offload: bool = False,
        diffusion_conv_direct: bool = False,  # NOT recommended - actually slower than default
        vae_conv_direct: bool = True,  # RECOMMENDED - 3x faster VAE decode with direct conv2d
        circular: bool = False,
        enable_flash_attention: bool = True,  # RECOMMENDED - faster with COOPMAT1 + reduces memory
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
            enable_vae_tiling: Process VAE in tiles to reduce memory usage
            enable_mmap: Memory-map model weights (can reduce RAM usage)
            keep_clip_on_cpu: Keep CLIP encoder in CPU (for low VRAM)
            enable_model_cpu_offload: Offload model weights to RAM (auto-load to VRAM as needed)
            diffusion_conv_direct: Use ggml_conv2d_direct in diffusion model (potentially faster)
            vae_conv_direct: Use ggml_conv2d_direct in VAE model (potentially faster)
            circular: Enable circular padding for seamless/tiling images
            enable_flash_attention: Use flash attention in diffusion model (FASTER with COOPMAT1 AND lower memory)
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
        self.enable_vae_tiling = enable_vae_tiling
        self.enable_mmap = enable_mmap
        self.keep_clip_on_cpu = keep_clip_on_cpu
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self.diffusion_conv_direct = diffusion_conv_direct
        self.vae_conv_direct = vae_conv_direct
        self.circular = circular
        self.enable_flash_attention = enable_flash_attention
        
        # Log initialization parameters
        logger.info(f"SDCppBackend init: vae_on_cpu={self.vae_on_cpu}, vae_tiling={self.enable_vae_tiling}, "
                   f"mmap={self.enable_mmap}, clip_cpu={self.keep_clip_on_cpu}, "
                   f"model_offload={self.enable_model_cpu_offload}")
        
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
        input_images: Optional[list] = None,
        strength: Optional[float] = None,
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
            input_images: Input images for img2img (not yet supported by sdcpp backend)
            strength: Denoising strength for img2img (not yet supported by sdcpp backend)
            
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
        
        # Load model-specific configuration if available
        model_config = self._load_model_config(model_path)
        
        # Apply defaults
        steps = steps or self.default_steps
        guidance_scale = guidance_scale or self.default_guidance_scale
        width = width or self.default_width
        height = height or self.default_height
        scheduler = scheduler or self.default_scheduler
        
        # Apply model-specific defaults (override generic defaults but not user-specified values)
        if model_config:
            defaults = model_config.get("defaults", {})
            if defaults.get("guidance_scale") and guidance_scale == self.default_guidance_scale:
                guidance_scale = defaults["guidance_scale"]
            if defaults.get("scheduler") and scheduler == self.default_scheduler:
                scheduler = defaults["scheduler"]
            if defaults.get("steps") and steps == self.default_steps:
                steps = defaults["steps"]
        
        # Map scheduler name
        sdcpp_scheduler = SCHEDULER_MAPPING.get(scheduler, "euler_a")
        if scheduler not in SCHEDULER_MAPPING:
            logger.warning(f"Unknown scheduler '{scheduler}', using 'euler_a'")
        
        # Generate unique output filename
        image_id = uuid.uuid4().hex[:12]
        output_path = self.images_dir / f"{image_id}.png"
        
        # Determine if this is a multi-component model (e.g., Qwen, FLUX)
        # that needs --diffusion-model instead of -m
        is_multicomponent = model_config is not None and (
            model_config.get("vae") or model_config.get("llm")
        )
        
        # Build command
        cmd = [str(self.sdcpp_binary)]
        
        if is_multicomponent:
            # Multi-component model: use --diffusion-model and separate component flags
            cmd.extend(["--diffusion-model", str(model_path)])
            
            # VAE path
            vae_path = model_config.get("vae")
            if vae_path:
                vae_full_path = self._resolve_auxiliary_path(model_path, vae_path)
                if vae_full_path:
                    cmd.extend(["--vae", str(vae_full_path)])
                else:
                    logger.warning(f"VAE file not found: {vae_path}")
            
            # LLM text encoder
            llm_path = model_config.get("llm")
            if llm_path:
                llm_full_path = self._resolve_auxiliary_path(model_path, llm_path)
                if llm_full_path:
                    cmd.extend(["--llm", str(llm_full_path)])
                else:
                    logger.warning(f"LLM encoder file not found: {llm_path}")
            
            # LLM vision encoder
            llm_vision_path = model_config.get("llm_vision")
            if llm_vision_path:
                llm_vision_full_path = self._resolve_auxiliary_path(model_path, llm_vision_path)
                if llm_vision_full_path:
                    cmd.extend(["--llm_vision", str(llm_vision_full_path)])
                else:
                    logger.warning(f"LLM vision encoder file not found: {llm_vision_path}")
            
            # CLIP vision encoder
            clip_vision_path = model_config.get("clip_vision")
            if clip_vision_path:
                clip_vision_full_path = self._resolve_auxiliary_path(model_path, clip_vision_path)
                if clip_vision_full_path:
                    cmd.extend(["--clip_vision", str(clip_vision_full_path)])
                else:
                    logger.warning(f"CLIP vision encoder file not found: {clip_vision_path}")
        else:
            # Standard single-file model
            cmd.extend(["-m", str(model_path)])
        
        # Core generation parameters
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
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # Input image for img2img / image editing
        if input_images and len(input_images) > 0:
            # Save the first input image to a temp file for sd-cli
            import tempfile
            input_img = input_images[0]
            temp_input = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=str(self.images_dir))
            input_img.save(temp_input.name)
            temp_input.close()
            
            # Use -r for reference image (Qwen image edit uses this)
            # Use -i for standard img2img init image
            if model_config and model_config.get("model_type", "").startswith("qwen"):
                cmd.extend(["-r", temp_input.name])
            else:
                cmd.extend(["-i", temp_input.name])
            
            # Denoising strength
            if strength is not None:
                cmd.extend(["--strength", str(strength)])
            
            logger.info(f"img2img mode: input image saved to {temp_input.name}")
        
        # Hardware optimization flags
        if self.vae_on_cpu:
            cmd.append("--vae-on-cpu")
        if self.enable_vae_tiling:
            cmd.append("--vae-tiling")
        if self.enable_mmap:
            cmd.append("--mmap")
        if self.keep_clip_on_cpu:
            cmd.append("--clip-on-cpu")
        if self.enable_model_cpu_offload:
            cmd.append("--offload-to-cpu")
        if self.diffusion_conv_direct:
            cmd.append("--diffusion-conv-direct")
        if self.vae_conv_direct:
            cmd.append("--vae-conv-direct")
        if self.circular:
            cmd.append("--circular")
        if self.enable_flash_attention:
            cmd.append("--diffusion-fa")
        
        # Model-specific extra flags from config
        if model_config:
            extra_flags = model_config.get("flags", [])
            if extra_flags:
                cmd.extend(extra_flags)
                logger.info(f"Added model-specific flags: {extra_flags}")
        
        # LoRA support check
        if lora_paths:
            logger.warning("LoRA support not yet implemented for sd.cpp backend")
        
        # Multi-image warning
        if num_images > 1:
            logger.warning(f"Requested {num_images} images, but sd.cpp backend only supports 1 per call")
        
        # Log command (sanitized)
        logger.info(f"Executing sd-cli: steps={steps}, scheduler={sdcpp_scheduler}, size={width}x{height}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Full command: {' '.join(cmd)}")
        
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
            logger.info("Starting sd-cli execution...")
            result = await loop.run_in_executor(None, _run_subprocess)
            logger.info(f"sd-cli completed with return code: {result.returncode}")
            
            # Check cancellation
            if cancellation_token and cancellation_token.is_cancelled():
                # Clean up output file if created
                if output_path.exists():
                    output_path.unlink()
                raise CancellationError("Generation cancelled")
            
            # Log stdout and stderr for debugging
            stdout = result.stdout.decode('utf-8', errors='replace')
            stderr = result.stderr.decode('utf-8', errors='replace')
            if stdout:
                logger.info(f"sd-cli stdout: {stdout[:500]}")
            if stderr:
                logger.info(f"sd-cli stderr: {stderr[:500]}")
            
            # Check exit code
            if result.returncode != 0:
                error_msg = stderr
                logger.error(f"sd-cli failed with code {result.returncode}: {error_msg}")
                raise RuntimeError(f"Image generation failed: {error_msg}")
            
            # Verify output file exists with multiple checks
            logger.info(f"Checking for output file: {output_path}")
            logger.info(f"Output path exists: {output_path.exists()}")
            logger.info(f"Output path is_file: {output_path.is_file() if output_path.exists() else 'N/A'}")
            
            # Try listing directory contents
            try:
                dir_contents = list(self.images_dir.glob(f"{image_id}*"))
                logger.info(f"Files matching {image_id}*: {dir_contents}")
            except Exception as e:
                logger.warning(f"Could not list directory: {e}")
            
            if not output_path.exists():
                logger.error(f"sd-cli succeeded but output file not found: {output_path}")
                logger.error(f"Directory {self.images_dir} contains: {list(self.images_dir.iterdir())[:10]}")
                raise RuntimeError("Image generation completed but output file not created")
            
            elapsed = time.time() - start_time
            
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
        
        finally:
            # Clean up temp input image file if created
            if input_images and len(input_images) > 0:
                try:
                    import os
                    if 'temp_input' in dir() and hasattr(temp_input, 'name') and os.path.exists(temp_input.name):
                        os.unlink(temp_input.name)
                        logger.debug(f"Cleaned up temp input image: {temp_input.name}")
                except Exception as cleanup_err:
                    logger.debug(f"Could not clean up temp input image: {cleanup_err}")
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get Vulkan GPU information.
        
        Returns:
            Dict with GPU details (name, type, driver, VRAM if available, gpu_available)
        """
        info = {
            "type": "vulkan",
            "backend": "stable-diffusion.cpp",
            "gpu_available": True,
            "stats_available": False,
            "memory_used": "0 GB",
            "memory_total": "0 GB",
            "utilization": 0.0,
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
                device_name = "Unknown Vulkan Device"
                for line in output.split('\n'):
                    if 'deviceName' in line or 'GPU' in line:
                        parts = line.split('=')
                        if len(parts) > 1:
                            device_name = parts[1].strip()
                            break
                
                info["name"] = device_name
                info["driver"] = "Vulkan"
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.debug("vulkaninfo not available, using generic Vulkan info")
            info["name"] = "Vulkan Device"
            info["driver"] = "Vulkan"
        
        # Try to get AMD GPU stats from multiple sources
        # Priority: sysfs (fastest) -> rocm-smi (if available)
        
        # Method 1: AMD sysfs (works on all AMD GPUs with kernel driver)
        try:
            gpu_cards = glob.glob('/sys/class/drm/card*/device/mem_info_vram_total')
            
            if gpu_cards:
                card_path = Path(gpu_cards[0]).parent
                
                # Read VRAM stats
                vram_total_path = card_path / 'mem_info_vram_total'
                vram_used_path = card_path / 'mem_info_vram_used'
                gpu_busy_path = card_path / 'gpu_busy_percent'
                
                if vram_total_path.exists() and vram_used_path.exists():
                    vram_total = int(vram_total_path.read_text().strip())
                    vram_used = int(vram_used_path.read_text().strip())
                    
                    info["memory_used"] = f"{vram_used / (1024**3):.1f} GB"
                    info["memory_total"] = f"{vram_total / (1024**3):.1f} GB"
                    info["utilization"] = vram_used / vram_total if vram_total > 0 else 0.0
                    info["stats_available"] = True
                    info["stats_source"] = "sysfs"
                    
                    # Try to get GPU busy percent if available
                    if gpu_busy_path.exists():
                        try:
                            gpu_busy = int(gpu_busy_path.read_text().strip())
                            info["gpu_busy_percent"] = gpu_busy
                        except (ValueError, IOError):
                            pass
                    
                    return info  # Success via sysfs
                        
        except (FileNotFoundError, ValueError, IOError, IndexError):
            logger.debug("AMD sysfs stats not available")
        
        # Method 2: rocm-smi (if installed)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showmeminfo", "vram", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
                check=False
            )
            
            if result.returncode == 0:
                import json as json_module
                data = json_module.loads(result.stdout.decode('utf-8'))
                
                # Parse ROCm memory info
                if isinstance(data, dict):
                    # Try to find first GPU card entry
                    for key, value in data.items():
                        if key.startswith('card'):
                            vram_used = value.get('VRAM Total Used Memory (B)', 0)
                            vram_total = value.get('VRAM Total Memory (B)', 0)
                            
                            if vram_total > 0:
                                info["memory_used"] = f"{vram_used / (1024**3):.1f} GB"
                                info["memory_total"] = f"{vram_total / (1024**3):.1f} GB"
                                info["utilization"] = vram_used / vram_total
                                info["stats_available"] = True
                                info["stats_source"] = "rocm-smi"
                                return info  # Success via rocm-smi
                            break
        except Exception as e:
            logger.debug(f"rocm-smi not available: {e}")
        
        return info
    
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
    def _load_model_config(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load model-specific configuration from a YAML file alongside the model.
        
        Looks for a .yaml file with the same stem as the model file.
        For example: qwen-image-edit-2511.gguf -> qwen-image-edit-2511.yaml
        
        The YAML config can specify:
            model_type: qwen_image_edit_2511
            vae: qwen_image_vae.safetensors
            llm: Qwen2.5-VL-7B-Instruct-Q8_0.gguf
            llm_vision: Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf
            clip_vision: clip_vision.safetensors
            flags:
              - --qwen-image-zero-cond-t
              - --flow-shift
              - "3"
            defaults:
              guidance_scale: 2.5
              scheduler: euler
              steps: 30
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Configuration dict or None if no config found
        """
        import yaml
        
        # Look for config with same stem
        config_path = model_path.with_suffix(".yaml")
        if not config_path.exists():
            # Also try .yml extension
            config_path = model_path.with_suffix(".yml")
        
        if not config_path.exists():
            # If model is inside a directory, check for config in the directory
            if model_path.parent != model_path.parent.parent:
                dir_config = model_path.parent / "model_config.yaml"
                if dir_config.exists():
                    config_path = dir_config
                else:
                    dir_config = model_path.parent / "model_config.yml"
                    if dir_config.exists():
                        config_path = dir_config
        
        if not config_path.exists():
            return None
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"Loaded model config from {config_path}: type={config.get('model_type', 'unknown')}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load model config {config_path}: {e}")
            return None
    
    def _resolve_auxiliary_path(self, model_path: Path, aux_filename: str) -> Optional[Path]:
        """
        Resolve path to an auxiliary model file (VAE, LLM encoder, etc.).
        
        Search order:
        1. Same directory as the main model file
        2. Parent directory (models root)
        3. Absolute path (if aux_filename is absolute)
        
        Args:
            model_path: Path to the main model file
            aux_filename: Filename or relative path of auxiliary file
            
        Returns:
            Resolved Path or None if not found
        """
        aux_path = Path(aux_filename)
        
        # If absolute path, use directly
        if aux_path.is_absolute():
            if aux_path.exists():
                return aux_path
            logger.warning(f"Auxiliary file not found at absolute path: {aux_path}")
            return None
        
        # Search in same directory as model
        candidate = model_path.parent / aux_filename
        if candidate.exists():
            logger.debug(f"Found auxiliary file: {candidate}")
            return candidate
        
        # Search in parent directory (models root)
        candidate = model_path.parent.parent / aux_filename
        if candidate.exists():
            logger.debug(f"Found auxiliary file in parent: {candidate}")
            return candidate
        
        # Search with glob for partial matches (e.g., "qwen_image_vae*")
        for pattern_dir in [model_path.parent, model_path.parent.parent]:
            matches = list(pattern_dir.glob(f"{aux_path.stem}*{aux_path.suffix}"))
            if matches:
                logger.debug(f"Found auxiliary file via glob: {matches[0]}")
                return matches[0]
        
        return None
    
    @staticmethod
    def get_backend_name() -> str:
        """
        Get human-readable backend name.
        
        Returns:
            Backend name string
        """
        return "Vulkan (stable-diffusion.cpp)"
