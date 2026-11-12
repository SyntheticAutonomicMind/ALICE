# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""Tests for torch.compile warmup and denoiser resolution.

Covers:
1. ``_denoiser_attr`` finds ``unet`` on UNet pipelines and ``transformer``
   on transformer pipelines (Qwen Image, FLUX).  Compiling only ``unet``
   silently skipped every transformer model.
2. ``warmup_compile`` is exposed through config, GeneratorService, and
   PyTorchBackend so codegen can be moved off the request path.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.backends.pytorch_backend import PyTorchBackend
from src.config import GenerationConfig


def _backend():
    """Build a backend without touching torch or the filesystem."""
    return PyTorchBackend(images_dir=Path("/tmp/alice-test-images"))


class TestDenoiserAttr:
    """The compiled module differs between pipeline families."""

    def test_returns_unet_for_unet_pipelines(self):
        backend = _backend()
        backend._pipeline = MagicMock(spec=["unet"])
        assert backend._denoiser_attr() == "unet"

    def test_returns_transformer_for_transformer_pipelines(self):
        backend = _backend()
        backend._pipeline = MagicMock(spec=["transformer"])
        assert backend._denoiser_attr() == "transformer"

    def test_prefers_unet_when_both_present(self):
        backend = _backend()
        backend._pipeline = MagicMock(spec=["unet", "transformer"])
        assert backend._denoiser_attr() == "unet"

    def test_returns_none_when_neither_present(self):
        backend = _backend()
        backend._pipeline = MagicMock(spec=["vae"])
        assert backend._denoiser_attr() is None

    def test_returns_none_without_pipeline(self):
        backend = _backend()
        backend._pipeline = None
        assert backend._denoiser_attr() is None


class TestWarmupConfig:
    """warmup_compile must be plumbed from config to the backend."""

    def test_config_defaults_to_enabled(self):
        assert GenerationConfig().warmup_compile is True

    def test_backend_accepts_warmup_flag(self):
        backend = PyTorchBackend(images_dir=Path("/tmp/alice-test-images"), warmup_compile=False)
        assert backend.warmup_compile is False

    def test_backend_defaults_to_enabled(self):
        assert _backend().warmup_compile is True

    def test_generator_service_forwards_flag(self):
        from src.generator import GeneratorService

        service = GeneratorService(images_dir=Path("/tmp/alice-test-images"), warmup_compile=False)
        assert service._backend_kwargs["warmup_compile"] is False


class TestWarmupSkipsWithoutCompile:
    """Warmup must be a no-op when nothing was compiled."""

    def test_noop_when_no_original_unet(self):
        backend = _backend()
        backend._pipeline = MagicMock()
        backend._original_unet = None
        # Must not raise and must not call the pipeline.
        backend._warmup_compiled_unet()
        backend._pipeline.assert_not_called()

    def test_noop_when_no_pipeline(self):
        backend = _backend()
        backend._pipeline = None
        backend._original_unet = MagicMock()
        backend._warmup_compiled_unet()
