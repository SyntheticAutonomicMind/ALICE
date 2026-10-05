# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 The ALICE Authors

"""
ALICE Generator Tests

Tests for the image generation engine.
Run with: pytest tests/test_generator.py -v

Tests exercise the current GeneratorService / PyTorchBackend API.
Some tests require diffusers to be installed.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Check if torch and diffusers are available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import diffusers
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False

# GeneratorService and PyTorchBackend require torch at import time
HAS_TORCH_AND_BACKEND = HAS_TORCH and HAS_DIFFUSERS


# ---------------------------------------------------------------------------
# GeneratorService wrapper tests (no backend needed for construction)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_generator_initialization():
    """Test GeneratorService initializes correctly with a backend."""
    from src.generator import GeneratorService

    generator = GeneratorService(
        images_dir=Path("./test_images"),
        backend_name="pytorch",
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )

    assert generator is not None
    assert generator._backend is not None
    assert generator.total_generations == 0


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_generator_delegates_to_backend():
    """GeneratorService delegates get_gpu_info to its backend."""
    from src.generator import GeneratorService

    generator = GeneratorService(
        images_dir=Path("./test_images"),
        backend_name="pytorch",
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )

    info = generator.get_gpu_info()

    assert "gpu_available" in info
    assert "memory_used" in info
    assert "memory_total" in info
    assert "utilization" in info
    assert "device" in info


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_generator_backend_properties():
    """GeneratorService exposes backend's current_model and is_model_loaded."""
    from src.generator import GeneratorService

    generator = GeneratorService(
        images_dir=Path("./test_images"),
        backend_name="pytorch",
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )

    # Initially no model loaded
    assert not generator.is_model_loaded
    assert generator.current_model is None


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_generator_total_generations():
    """Test generator tracks total generations."""
    from src.generator import GeneratorService

    generator = GeneratorService(
        images_dir=Path("./test_images"),
        backend_name="pytorch",
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )

    assert generator.total_generations == 0


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_generator_generate_alias():
    """GeneratorService.generate is an alias for generate_image."""
    from src.generator import GeneratorService

    generator = GeneratorService(
        images_dir=Path("./test_images"),
        backend_name="pytorch",
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )

    assert callable(generator.generate)
    assert generator.generate.__name__ == "generate"


# ---------------------------------------------------------------------------
# PyTorchBackend tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_pytorch_backend_device():
    """PyTorchBackend detects a device."""
    from src.backends.pytorch_backend import PyTorchBackend

    backend = PyTorchBackend(
        images_dir=Path("./test_images"),
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )

    assert backend._device in ("cuda", "mps", "cpu")


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_pytorch_backend_no_model_loaded():
    """PyTorchBackend reports no model loaded initially."""
    from src.backends.pytorch_backend import PyTorchBackend

    backend = PyTorchBackend(
        images_dir=Path("./test_images"),
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )

    assert not backend.is_model_loaded
    assert backend.current_model is None


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_pytorch_backend_scheduler_classes():
    """Test scheduler classes are populated after diffusers import."""
    from src.backends.pytorch_backend import _import_diffusers, _scheduler_classes

    _import_diffusers()

    # Should have common schedulers
    assert "euler" in _scheduler_classes
    assert "euler_a" in _scheduler_classes
    assert "ddim" in _scheduler_classes
    assert "dpm++_sde_karras" in _scheduler_classes


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_pytorch_backend_scheduler_map_completeness():
    """Test scheduler map has all documented schedulers."""
    from src.backends.pytorch_backend import _import_diffusers, _scheduler_classes

    _import_diffusers()

    documented_schedulers = [
        "euler",
        "euler_a",
        "ddim",
        "pndm",
        "lms",
        "dpm++_karras",
        "dpm++_sde_karras",
    ]

    for scheduler in documented_schedulers:
        assert scheduler in _scheduler_classes, f"Missing scheduler: {scheduler}"


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_pytorch_backend_get_scheduler():
    """Test scheduler creation function."""
    from src.backends.pytorch_backend import _import_diffusers, _get_scheduler

    _import_diffusers()

    # Get a sample scheduler config from diffusers
    from diffusers import EulerDiscreteScheduler
    sample_config = EulerDiscreteScheduler.from_config({
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
    }).config

    scheduler = _get_scheduler("euler", sample_config)
    assert scheduler is not None


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_pytorch_backend_get_scheduler_invalid():
    """Test scheduler creation with invalid name raises error."""
    from src.backends.pytorch_backend import _import_diffusers, _get_scheduler

    _import_diffusers()

    with pytest.raises(ValueError) as exc_info:
        _get_scheduler("invalid_scheduler_name", {})

    assert "Unknown scheduler" in str(exc_info.value)


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_detect_pipeline_class():
    """Test pipeline class detection function."""
    from src.backends.pytorch_backend import _detect_pipeline_class
    import tempfile
    import json

    # Test with non-existent path
    pipeline_class, model_type = _detect_pipeline_class(Path("/nonexistent"))
    assert pipeline_class == "StableDiffusionPipeline"
    assert model_type == "sd15"

    # Test with directory containing model_index.json
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir)
        model_index = {"_class_name": "StableDiffusionXLPipeline"}
        with open(model_path / "model_index.json", "w") as f:
            json.dump(model_index, f)

        pipeline_class, model_type = _detect_pipeline_class(model_path)
        assert "sdxl" in model_type.lower() or "xl" in pipeline_class.lower()


def test_image_path_generation():
    """Test generated image path format."""
    import uuid
    from pathlib import Path

    # Simulate image path generation logic
    images_dir = Path("./images")
    image_id = uuid.uuid4().hex
    image_path = images_dir / f"{image_id}.png"

    assert image_path.suffix == ".png"
    assert len(image_id) == 32  # UUID hex length


@pytest.mark.skipif(not HAS_TORCH_AND_BACKEND, reason="torch/diffusers not installed")
def test_pytorch_backend_gpu_info():
    """Test PyTorchBackend get_gpu_info returns expected keys."""
    from src.backends.pytorch_backend import PyTorchBackend

    backend = PyTorchBackend(
        images_dir=Path("./test_images"),
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )

    info = backend.get_gpu_info()

    assert "device" in info
    assert "gpu_available" in info
    assert "memory_used" in info
    assert "memory_total" in info
    assert "utilization" in info
    assert "stats_available" in info
