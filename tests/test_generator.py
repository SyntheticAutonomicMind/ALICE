# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 The ALICE Authors

"""
ALICE Generator Tests

Tests for the image generation engine.
Run with: pytest tests/test_generator.py -v

Note: These tests don't require a GPU or actual models.
They test the generator infrastructure and error handling.
Some tests require diffusers to be installed.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Check if diffusers is available
try:
    import diffusers
    HAS_DIFFUSERS = True
except ImportError:
    HAS_DIFFUSERS = False


def test_generator_initialization():
    """Test generator initializes correctly."""
    from src.generator import GeneratorService
    
    generator = GeneratorService(
        images_dir=Path("./test_images"),
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )
    
    assert generator is not None
    assert generator.default_steps == 25
    assert generator.default_guidance_scale == 7.5
    assert generator.default_scheduler == "euler_a"
    assert generator.default_width == 512
    assert generator.default_height == 512


def test_generator_device_property():
    """Test generator has device property."""
    from src.generator import GeneratorService
    
    generator = GeneratorService(
        images_dir=Path("./test_images"),
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )
    
    # Should have _device attribute (set in __init__)
    assert hasattr(generator, "_device")
    # Device should be cuda, mps, or cpu
    assert generator._device in ["cuda", "mps", "cpu"]


def test_generator_no_model_loaded():
    """Test generator reports no model loaded initially."""
    from src.generator import GeneratorService
    
    generator = GeneratorService(
        images_dir=Path("./test_images"),
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )
    
    assert not generator.is_model_loaded
    assert generator.current_model is None


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_generator_scheduler_classes_after_import():
    """Test scheduler classes are populated after diffusers import."""
    from src.generator import _import_diffusers, _scheduler_classes
    
    # Import diffusers
    _import_diffusers()
    
    # Should have common schedulers
    assert "euler" in _scheduler_classes
    assert "euler_a" in _scheduler_classes
    assert "ddim" in _scheduler_classes
    assert "dpm++_sde_karras" in _scheduler_classes


def test_generator_total_generations():
    """Test generator tracks total generations."""
    from src.generator import GeneratorService
    
    generator = GeneratorService(
        images_dir=Path("./test_images"),
        default_steps=25,
        default_guidance_scale=7.5,
        default_scheduler="euler_a",
        default_width=512,
        default_height=512,
    )
    
    # Should have total_generations attribute
    assert hasattr(generator, "total_generations")
    assert generator.total_generations == 0


def test_generator_gpu_info_method():
    """Test generator has get_gpu_info method."""
    from src.generator import GeneratorService
    
    generator = GeneratorService(
        images_dir=Path("./test_images"),
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


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_generation_params_class():
    """Test GenerationParams dataclass exists and has fields."""
    from src.generator import GenerationParams
    
    params = GenerationParams(
        model_path=Path("./test"),
        prompt="test prompt",
    )
    
    assert params.prompt == "test prompt"
    assert params.negative_prompt == ""
    assert params.steps == 25


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_generation_params_defaults():
    """Test GenerationParams has correct default values."""
    from src.generator import GenerationParams
    
    params = GenerationParams(
        model_path=Path("./test"),
        prompt="test",
    )
    
    # Check defaults
    assert params.steps == 25
    assert params.guidance_scale == 7.5
    assert params.width == 512
    assert params.height == 512
    assert params.seed is None
    assert params.scheduler == "dpm++_sde_karras"


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_generation_result_class():
    """Test GenerationResult dataclass exists."""
    from src.generator import GenerationResult
    
    result = GenerationResult(
        image_path=Path("./test.png"),
        image_url="http://localhost/test.png",
        seed=42,
        prompt="test",
        steps=25,
        guidance_scale=7.5,
        scheduler="euler",
        width=512,
        height=512,
        generation_time=1.5,
    )
    
    assert result.image_url == "http://localhost/test.png"
    assert result.seed == 42
    assert result.generation_time == 1.5


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


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_scheduler_map_completeness():
    """Test scheduler map has all documented schedulers."""
    from src.generator import _import_diffusers, _scheduler_classes
    
    # Import diffusers first
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


def test_detect_pipeline_class():
    """Test pipeline class detection function."""
    from src.generator import _detect_pipeline_class
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


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_get_scheduler_function():
    """Test scheduler creation function."""
    from src.generator import _import_diffusers, _get_scheduler, _scheduler_classes
    
    # Import diffusers first
    _import_diffusers()
    
    # Get a sample scheduler config from diffusers
    from diffusers import EulerDiscreteScheduler
    sample_config = EulerDiscreteScheduler.from_config({
        "num_train_timesteps": 1000,
        "beta_start": 0.00085,
        "beta_end": 0.012,
    }).config
    
    # Should create scheduler
    scheduler = _get_scheduler("euler", sample_config)
    assert scheduler is not None


@pytest.mark.skipif(not HAS_DIFFUSERS, reason="diffusers not installed")
def test_get_scheduler_invalid_name():
    """Test scheduler creation with invalid name raises error."""
    from src.generator import _import_diffusers, _get_scheduler
    
    # Import diffusers first
    _import_diffusers()
    
    with pytest.raises(ValueError) as exc_info:
        _get_scheduler("invalid_scheduler_name", {})
    
    assert "Unknown scheduler" in str(exc_info.value)
