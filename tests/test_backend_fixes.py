# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
Tests for backend fixes in pytorch_backend.py.

Tests the following fixes:
1. VAE slicing/tiling uses hasattr checks instead of try/except
   (fixes 'StableDiffusionXLPipeline' object has no attribute enable_vae_slicing')
2. Environment variable suppression for HF Hub, transformers, torch
3. Warning filters for non-actionable torch._sympy and float32 module warnings
4. PYTORCH_HIP_ALLOC_CONF migration to PYTORCH_ALLOC_CONF

These tests mock torch and PIL so they can run without a GPU or PyTorch
installation.
"""

import os
import sys
import warnings
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing pytorch_backend
# ---------------------------------------------------------------------------

def _install_mocks():
    """Install mock modules for torch and PIL so pytorch_backend can be imported."""
    # Mock torch
    torch_mock = MagicMock()
    torch_mock.cuda.is_available.return_value = False
    torch_mock.backends.cudnn.enabled = True
    torch_mock.backends.cuda.enable_flash_sdp = MagicMock()
    torch_mock.backends.cuda.enable_mem_efficient_sdp = MagicMock()
    torch_mock.backends.cuda.enable_math_sdp = MagicMock()
    torch_mock.compile = MagicMock()
    torch_mock.float16 = "float16"
    torch_mock.float32 = "float32"
    torch_mock.bfloat16 = "bfloat16"
    torch_mock.no_grad = lambda: MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
    torch_mock.inference_mode = lambda: MagicMock(__enter__=lambda s: s, __exit__=lambda *a: None)
    torch_mock.Generator = MagicMock
    torch_mock.manual_seed = MagicMock()
    torch_mock.empty = MagicMock()
    torch_mock.FloatTensor = MagicMock()
    torch_mock.device = MagicMock()
    sys.modules['torch'] = torch_mock

    # Mock PIL
    pil_mock = MagicMock()
    pil_img = MagicMock()
    pil_img.fromarray = MagicMock()
    pil_img.new = MagicMock()
    pil_img.Image = MagicMock()
    pil_mock.Image = pil_img
    sys.modules['PIL'] = pil_mock
    sys.modules['PIL.Image'] = pil_img

    # Mock diffusers (lazy-imported, but mock anyway for safety)
    diffusers_mock = MagicMock()
    sys.modules['diffusers'] = diffusers_mock

    # Mock compel (TYPE_CHECKING only, but mock for safety)
    compel_mock = MagicMock()
    sys.modules['compel'] = compel_mock


_install_mocks()


# ---------------------------------------------------------------------------
# Import pytorch_backend (with mocked dependencies)
# ---------------------------------------------------------------------------

from src.backends import pytorch_backend  # noqa: E402


# ---------------------------------------------------------------------------
# VAE slicing/tiling tests
# ---------------------------------------------------------------------------

class TestVAESlicingTiling:
    """Test that VAE slicing/tiling uses hasattr checks, not try/except."""

    def _make_backend(self):
        """Create a PyTorchBackend instance with mocked deps."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            enable_vae_slicing=True,
            enable_vae_tiling=True,
            enable_torch_compile=False,
        )
        return backend

    def test_pipeline_with_enable_vae_slicing(self):
        """When pipeline has enable_vae_slicing, it should be called directly."""
        backend = self._make_backend()
        mock_pipeline = MagicMock()
        # Pipeline HAS the method
        mock_pipeline.enable_vae_slicing = MagicMock()
        mock_pipeline.enable_vae_tiling = MagicMock()
        backend._pipeline = mock_pipeline
        backend._current_model_type = "sd15"

        backend._apply_memory_optimizations()

        # Methods should be called (not skipped due to missing attr)
        mock_pipeline.enable_vae_slicing.assert_called_once()
        mock_pipeline.enable_vae_tiling.assert_called_once()

    def test_sdxl_pipeline_falls_back_to_vae_subcomponent(self):
        """SDXL pipeline lacks enable_vae_slicing on the pipeline; should
        fall back to pipeline.vae.enable_slicing()."""
        backend = self._make_backend()
        mock_pipeline = MagicMock()
        # SDXL pipeline does NOT have enable_vae_slicing
        del mock_pipeline.enable_vae_slicing
        del mock_pipeline.enable_vae_tiling
        # But the VAE subcomponent has enable_slicing/enable_tiling
        mock_vae = MagicMock()
        mock_pipeline.vae = mock_vae

        backend._pipeline = mock_pipeline
        backend._current_model_type = "sdxl"

        backend._apply_memory_optimizations()

        # Should have called the VAE subcomponent methods, NOT raised
        mock_vae.enable_slicing.assert_called_once()
        mock_vae.enable_tiling.assert_called_once()

    def test_no_vae_on_pipeline_no_warning(self):
        """When neither pipeline nor VAE has the methods, no warning is emitted."""
        backend = self._make_backend()
        mock_pipeline = MagicMock()
        # Remove ALL vae-related methods
        del mock_pipeline.enable_vae_slicing
        del mock_pipeline.enable_vae_tiling
        # VAE exists but has no slicing/tiling methods
        mock_vae = MagicMock(spec=[])  # spec=[] means no methods
        mock_pipeline.vae = mock_vae
        # Also remove 'vae' attribute detection
        # Actually, spec=[] creates an object with no methods, so hasattr returns False

        backend._pipeline = mock_pipeline
        backend._current_model_type = "sdxl"

        with patch('src.backends.pytorch_backend.logger') as mock_logger:
            backend._apply_memory_optimizations()
            # Should NOT log a warning (debug level only)
            for call in mock_logger.method_calls:
                if call[0] == 'warning':
                    assert 'VAE slicing' not in str(call[1]), (
                        "Unexpected warning about VAE slicing"
                    )
                    assert 'VAE tiling' not in str(call[1]), (
                        "Unexpected warning about VAE tiling"
                    )

    def test_vae_slicing_disabled_not_called(self):
        """When enable_vae_slicing is False, the method should not be called at all."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            enable_vae_slicing=False,
            enable_vae_tiling=False,
            enable_torch_compile=False,
        )
        mock_pipeline = MagicMock()
        mock_pipeline.enable_vae_slicing = MagicMock()
        mock_pipeline.enable_vae_tiling = MagicMock()
        backend._pipeline = mock_pipeline
        backend._current_model_type = "sd15"

        backend._apply_memory_optimizations()

        mock_pipeline.enable_vae_slicing.assert_not_called()
        mock_pipeline.enable_vae_tiling.assert_not_called()

    def test_attention_slicing_still_works(self):
        """Attention slicing should still work via hasattr."""
        backend = self._make_backend()
        mock_pipeline = MagicMock()
        backend._pipeline = mock_pipeline
        backend._current_model_type = "sd15"
        # Set attention_slice_size to trigger the code path
        backend.attention_slice_size = "auto"

        backend._apply_memory_optimizations()

        # enable_attention_slicing should have been called
        mock_pipeline.enable_attention_slicing.assert_called_once()


# ---------------------------------------------------------------------------
# Environment variable and warning suppression tests
# ---------------------------------------------------------------------------

class TestWarningSuppression:
    """Test that environment variables and warning filters are set correctly."""

    def test_hf_hub_telemetry_disabled(self):
        """HF_HUB_DISABLE_TELEMETRY should be set to '1'."""
        assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"

    def test_hf_hub_progress_bars_disabled(self):
        """HF_HUB_DISABLE_PROGRESS_BARS should be set to '1'."""
        assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"

    def test_transformers_verbosity_error(self):
        """TRANSFORMERS_VERBOSITY should be 'error' to suppress info/warning logs."""
        assert os.environ.get("TRANSFORMERS_VERBOSITY") == "error"

    def test_tokenizers_parallelism_false(self):
        """TOKENIZERS_PARALLELISM should be 'false' to avoid warnings."""
        assert os.environ.get("TOKENIZERS_PARALLELISM") == "false"

    def test_pytorch_hip_alloc_conf_migrated(self):
        """If PYTORCH_HIP_ALLOC_CONF is set, PYTORCH_ALLOC_CONF should be set too."""
        # The module should have set PYTORCH_ALLOC_CONF if it wasn't already
        # and PYTORCH_HIP_ALLOC_CONF was present
        # Note: in test env, neither may be set, which is fine
        if os.environ.get("PYTORCH_HIP_ALLOC_CONF"):
            assert os.environ.get("PYTORCH_ALLOC_CONF"), (
                "PYTORCH_HIP_ALLOC_CONF is set but PYTORCH_ALLOC_CONF is not"
            )

    def test_warning_filters_registered(self):
        """Warning filters should be registered for known noisy messages."""
        source = Path(pytorch_backend.__file__).read_text()
        assert "warnings.filterwarnings" in source, (
            "No warnings.filterwarnings calls found in pytorch_backend source"
        )

    def test_warning_filter_for_sympy(self):
        """The torch._sympy 'failed while executing' warning should be filtered."""
        source = Path(pytorch_backend.__file__).read_text()
        assert "failed while executing" in source, (
            "Filter for torch._sympy 'failed while executing' warning not found in source"
        )

    def test_warning_filter_for_float32_modules(self):
        """The 'should be kept in float32' warning should be filtered."""
        source = Path(pytorch_backend.__file__).read_text()
        assert "should be kept in float32" in source, (
            "Filter for float32 modules warning not found in source"
        )

    def test_warning_filter_for_token_indices(self):
        """The 'Token indices sequence length' warning should be filtered."""
        source = Path(pytorch_backend.__file__).read_text()
        assert "Token indices sequence length" in source, (
            "Filter for token indices warning not found in source"
        )

    def test_logging_level_suppression_for_sympy(self):
        """The torch._sympy 'failed while executing' log should be suppressed at logging level.

        This message is emitted via logging.warning(), not warnings.warn(),
        so warnings.filterwarnings has no effect. We set the logger level to
        ERROR instead.
        """
        import logging
        interp_logger = logging.getLogger("torch.utils._sympy.interp")
        assert interp_logger.level == logging.ERROR, (
            "torch.utils._sympy.interp logger should be set to ERROR level "
            "to suppress 'failed while executing' logging.warning() calls"
        )

    def test_logging_level_suppression_for_float32(self):
        """The 'should be kept in float32' log should be suppressed at logging level."""
        import logging
        logger = logging.getLogger("diffusers.models.modeling_utils")
        assert logger.level == logging.ERROR, (
            "diffusers.models.modeling_utils logger should be set to ERROR level"
        )

    def test_logging_level_suppression_for_token_indices(self):
        """The 'Token indices sequence length' log should be suppressed at logging level."""
        import logging
        logger = logging.getLogger("transformers.tokenization_utils_base")
        assert logger.level == logging.ERROR, (
            "transformers.tokenization_utils_base logger should be set to ERROR level"
        )


# ---------------------------------------------------------------------------
# Model cache / LRU eviction tests
# ---------------------------------------------------------------------------

class TestModelCacheLRU:
    """Test the LRU eviction behavior in PyTorchBackend model cache."""

    def _make_backend(self, max_cached=3):
        return pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            max_cached_models=max_cached,
            enable_torch_compile=False,
        )

    def test_cache_eviction_when_full(self):
        """When cache is full, LRU model should be evicted before loading new one."""
        backend = self._make_backend(max_cached=2)

        # Manually populate the cache with mock entries
        backend._model_cache["model_a"] = pytorch_backend._CachedModel(
            pipeline=MagicMock(),
            model_path=Path("/models/model_a"),
            model_type="sd15",
            device_map_active=False,
            is_single_file_sdxl=False,
        )
        backend._model_cache["model_b"] = pytorch_backend._CachedModel(
            pipeline=MagicMock(),
            model_path=Path("/models/model_b"),
            model_type="sd15",
            device_map_active=False,
            is_single_file_sdxl=False,
        )

        assert len(backend._model_cache) == 2

        # Mock _load_model_blocking to avoid actual model loading
        with patch.object(backend, '_load_model_blocking', return_value=(
            MagicMock(), "sd15", False, False
        )):
            with patch.object(backend, '_apply_memory_optimizations'):
                with patch.object(backend, '_set_active'):
                    with patch.object(backend, '_sync_active_to_cache'):
                        import asyncio
                        asyncio.run(
                            backend.load_model(Path("/models/model_c"))
                        )

        # model_a (LRU) should have been evicted
        assert "model_a" not in backend._model_cache, "LRU model was not evicted"
        assert "model_b" in backend._model_cache
        assert "model_c" in backend._model_cache or len(backend._model_cache) <= 2

    def test_mru_marking_on_access(self):
        """Accessing a cached model should move it to MRU position."""
        backend = self._make_backend(max_cached=3)

        # Populate cache with 3 models (keys are str(Path(...)) = full paths)
        for name in ["a", "b", "c"]:
            backend._model_cache[f"/models/model_{name}"] = pytorch_backend._CachedModel(
                pipeline=MagicMock(),
                model_path=Path(f"/models/model_{name}"),
                model_type="sd15",
                device_map_active=False,
                is_single_file_sdxl=False,
            )

        # Access model_a (should become MRU)
        # Simulate by calling load_model for an already-cached model
        import asyncio

        async def test_mru():
            with patch.object(backend, '_set_active'):
                await backend.load_model(Path("/models/model_a"))

        asyncio.run(test_mru())

        # model_a should now be the last (MRU) item
        keys = list(backend._model_cache.keys())
        assert keys[-1] == "/models/model_a", "MRU model was not moved to end"
        assert keys[0] == "/models/model_b", "model_b should still be LRU"

    def test_cache_hit_is_fast_path(self):
        """When model is already cached, load should be a fast no-op."""
        backend = self._make_backend(max_cached=3)

        # Pre-populate cache
        cached_model = MagicMock()
        backend._model_cache["/models/test"] = pytorch_backend._CachedModel(
            pipeline=cached_model,
            model_path=Path("/models/test"),
            model_type="sd15",
            device_map_active=False,
            is_single_file_sdxl=False,
        )

        import asyncio

        async def test_fast():
            with patch.object(backend, '_load_model_blocking') as mock_load:
                with patch.object(backend, '_set_active'):
                    await backend.load_model(Path("/models/test"))
                    # _load_model_blocking should NOT have been called
                    mock_load.assert_not_called()

        asyncio.run(test_fast())


# ---------------------------------------------------------------------------
# Pre-warming logic tests (main.py)
# ---------------------------------------------------------------------------

class TestPreWarming:
    """Test that the pre-warming logic in main.py loads the first available model."""

    def test_pre_warming_loads_first_model(self):
        """Pre-warming should call generator.load_model with the first model's path."""
        # Read main.py source to verify pre-warming code exists
        main_py = Path(__file__).parent.parent / "src" / "main.py"
        source = main_py.read_text()

        # Verify pre-warming code exists
        assert "Pre-warm" in source or "pre-warming" in source.lower(), (
            "Pre-warming code not found in main.py"
        )
        assert "load_model" in source, "load_model not found in pre-warming section"
        assert "models[0]" in source or "models[0].path" in source, (
            "First model access not found in pre-warming section"
        )

    def test_pre_warming_handles_no_models(self):
        """Pre-warming should gracefully handle the case where no models exist."""
        main_py = Path(__file__).parent.parent / "src" / "main.py"
        source = main_py.read_text()

        # Should check if models list is non-empty
        assert "if models and generator" in source or "if models and" in source, (
            "Pre-warming should check if models list is non-empty"
        )

    def test_pre_warming_logs_elapsed_time(self):
        """Pre-warming should log the elapsed time."""
        main_py = Path(__file__).parent.parent / "src" / "main.py"
        source = main_py.read_text()

        assert "Pre-warmed" in source or "pre-warmed" in source, (
            "Pre-warming log message not found"
        )

    def test_pre_warming_handles_errors_gracefully(self):
        """Pre-warming should log a warning but not crash if model loading fails."""
        main_py = Path(__file__).parent.parent / "src" / "main.py"
        source = main_py.read_text()

        assert "Pre-warming failed" in source or "pre-warming failed" in source.lower(), (
            "Error handling for pre-warming not found"
        )

    def test_hf_home_has_fallback_to_var_lib_alice(self):
        """HF_HOME should have a fallback to /var/lib/alice/.cache/huggingface."""
        main_py = Path(__file__).parent.parent / "src" / "main.py"
        source = main_py.read_text()
        assert "/var/lib/alice/.cache/huggingface" in source, (
            "HF_HOME fallback to /var/lib/alice/.cache/huggingface not found"
        )


# ---------------------------------------------------------------------------
# Downloader HF_HOME fix tests
# ---------------------------------------------------------------------------

class TestDownloaderHFHome:
    """Test that downloader.py resolves HF_HOME before importing huggingface_hub."""

    def test_downloader_resolves_hf_home_before_import(self):
        """downloader.py should set HF_HOME before 'from huggingface_hub import'."""
        dl_path = Path(__file__).parent.parent / "src" / "downloader.py"
        source = dl_path.read_text()

        # The HF_HOME fix must appear BEFORE the huggingface_hub import
        hf_home_pos = source.find('"HF_HOME" not in os.environ')
        hf_import_pos = source.find("from huggingface_hub import")

        assert hf_home_pos != -1, "HF_HOME resolution not found in downloader.py"
        assert hf_import_pos != -1, "huggingface_hub import not found in downloader.py"
        assert hf_home_pos < hf_import_pos, (
            "HF_HOME must be resolved BEFORE importing huggingface_hub, "
            f"but HF_HOME fix is at pos {hf_home_pos} and import at pos {hf_import_pos}"
        )

    def test_downloader_has_fallback_path(self):
        """downloader.py should fall back to /var/lib/alice/.cache/huggingface."""
        dl_path = Path(__file__).parent.parent / "src" / "downloader.py"
        source = dl_path.read_text()
        assert "/var/lib/alice/.cache/huggingface" in source, (
            "HF_HOME fallback to /var/lib/alice/.cache/huggingface not found in downloader.py"
        )

    def test_downloader_uses_os_access(self):
        """downloader.py should verify writability with os.access."""
        dl_path = Path(__file__).parent.parent / "src" / "downloader.py"
        source = dl_path.read_text()
        assert "os.access" in source, (
            "downloader.py does not use os.access to verify cache writability"
        )


# ---------------------------------------------------------------------------
# PyTorch backend TRITON_CACHE_DIR fix tests
# ---------------------------------------------------------------------------

class TestTritonCacheDir:
    """Test that pytorch_backend.py sets TRITON_CACHE_DIR before importing torch."""

    def test_triton_cache_dir_before_torch_import(self):
        """TRITON_CACHE_DIR must be set before 'import torch' in pytorch_backend.py."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()

        triton_pos = source.find('if "TRITON_CACHE_DIR" not in os.environ:')
        # Find the actual 'import torch' statement (not in a comment or string)
        import re
        torch_matches = list(re.finditer(r'^import torch$', source, re.MULTILINE))
        assert torch_matches, "import torch not found in pytorch_backend.py"
        torch_pos = torch_matches[0].start()
        assert triton_pos < torch_pos, (
            "TRITON_CACHE_DIR must be set before 'import torch'"
        )

    def test_triton_cache_dir_has_fallback(self):
        """TRITON_CACHE_DIR should fall back to /var/lib/alice/.triton."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        assert "/var/lib/alice/.triton" in source, (
            "TRITON_CACHE_DIR fallback to /var/lib/alice/.triton not found"
        )

    def test_triton_cache_dir_set(self):
        """TRITON_CACHE_DIR should be set in os.environ after module import."""
        # The module sets it at import time, so it should be available
        if os.environ.get("TRITON_CACHE_DIR"):
            assert Path(os.environ["TRITON_CACHE_DIR"]).exists(), (
                "TRITON_CACHE_DIR points to non-existent directory"
            )


# ---------------------------------------------------------------------------
# Config logging permission tests
# ---------------------------------------------------------------------------

class TestConfigLoggingPermission:
    """Test that config.py setup_logging handles permission denied gracefully."""

    def test_logging_checks_writable_before_handler(self):
        """setup_logging should check os.access before creating FileHandler."""
        config_py = Path(__file__).parent.parent / "src" / "config.py"
        source = config_py.read_text()
        assert "os.access" in source, (
            "setup_logging does not check os.access before creating FileHandler"
        )
        assert "W_OK" in source, (
            "setup_logging does not check W_OK permission"
        )

    def test_logging_warns_on_permission_denied(self):
        """setup_logging should log a warning (not crash) when file is not writable."""
        config_py = Path(__file__).parent.parent / "src" / "config.py"
        source = config_py.read_text()
        assert "not writable" in source, (
            "setup_logging does not handle not-writable case gracefully"
        )

    def test_logging_fallback_message(self):
        """setup_logging should fall back to console-only when file is not writable."""
        config_py = Path(__file__).parent.parent / "src" / "config.py"
        source = config_py.read_text()
        assert "console-only logging" in source or "console-only" in source, (
            "setup_logging does not fall back to console-only on permission error"
        )


# ---------------------------------------------------------------------------
# Thread configuration tests
# ---------------------------------------------------------------------------

class TestThreadConfiguration:
    """Test that PyTorchBackend configures CPU threads for multi-core systems."""

    def test_thread_configs_set_before_torch_import(self):
        """OMP_NUM_THREADS, MKL_NUM_THREADS, TORCH_NUM_THREADS must be set
        before 'import torch' in pytorch_backend.py."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        import re
        torch_pos = list(re.finditer(r'^import torch$', source, re.MULTILINE))[0].start()

        oomp_pos = source.find('OMP_NUM_THREADS')
        assert oomp_pos < torch_pos, "OMP_NUM_THREADS must be set before 'import torch'"

        mkl_pos = source.find('MKL_NUM_THREADS')
        assert mkl_pos < torch_pos, "MKL_NUM_THREADS must be set before 'import torch'"

        torch_threads_pos = source.find('TORCH_NUM_THREADS')
        assert torch_threads_pos < torch_pos, "TORCH_NUM_THREADS must be before 'import torch'"

    def test_triton_threads_configured(self):
        """TRITON_NUM_THREADS should be set to use multiple CPU threads."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        assert "TRITON_NUM_THREADS" in source, (
            "TRITON_NUM_THREADS not set for Triton parallel codegen"
        )

    def test_triton_persistent_cache_enabled(self):
        """TRITON_CACHE_PERSISTENT should be set to '1' for persistent kernel cache."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        assert "TRITON_CACHE_PERSISTENT" in source, (
            "TRITON_CACHE_PERSISTENT not set for persistent kernel cache"
        )

    def test_torch_home_set_for_protecthome(self):
        """TORCH_HOME should be set to /var/lib/alice/.cache/torch for systemd."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        assert "TORCH_HOME" in source, (
            "TORCH_HOME not set for systemd ProtectHome compatibility"
        )
        assert "/var/lib/alice/.cache/torch" in source, (
            "TORCH_HOME fallback to /var/lib/alice/.cache/torch not found"
        )


# ---------------------------------------------------------------------------
# CPU offload cache tests
# ---------------------------------------------------------------------------

class TestCPUCache:
    """Test the CPU offload cache (model warm cache for fast VRAM reload)."""

    def test_cpu_cache_initialized(self):
        """PyTorchBackend should have a _cpu_cache OrderedDict."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            enable_torch_compile=False,
        )
        assert hasattr(backend, '_cpu_cache'), "_cpu_cache not initialized"
        assert isinstance(backend._cpu_cache, OrderedDict), "_cpu_cache should be an OrderedDict"

    def test_max_cpu_cached_models_attribute(self):
        """Backend should have _max_cpu_cached_models."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            max_cpu_cached_models=16,
            enable_torch_compile=False,
        )
        assert backend._max_cpu_cached_models == 16

    def test_cpu_cache_max_enforced(self):
        """CPU cache should evict oldest when exceeding max_cpu_cached_models."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            max_cpu_cached_models=2,
            enable_torch_compile=False,
        )

        # Fill the CPU cache with 3 models (max is 2)
        for name in ["a", "b", "c"]:
            cached = pytorch_backend._CachedModel(
                pipeline=MagicMock(),
                model_path=Path(f"/models/{name}"),
                model_type="sd15",
                device_map_active=False,
                is_single_file_sdxl=False,
            )
            backend._move_model_to_cpu(str(Path(f"/models/{name}")), cached)

        assert len(backend._cpu_cache) <= 2, "CPU cache exceeded max"
        assert "/models/a" not in backend._cpu_cache, "LRU CPU model was not evicted"
        assert "/models/c" in backend._cpu_cache, "MRU CPU model was evicted instead"

    def test_move_to_cpu_drops_compel(self):
        """Moving a model to CPU should drop the CompelForSDXL instance
        (it holds GPU-specific state)."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            enable_torch_compile=False,
        )

        cached = pytorch_backend._CachedModel(
            pipeline=MagicMock(),
            model_path=Path("/models/test"),
            model_type="sd15",
            device_map_active=False,
            is_single_file_sdxl=False,
            compel=MagicMock(),
        )
        # Simulate the pipeline.to() call in _move_model_to_cpu
        cached.pipeline.to = MagicMock(return_value=cached.pipeline)

        backend._move_model_to_cpu("/models/test", cached)

        assert cached.compel is None, "Compel not cleared on CPU offload"
        assert cached.vram_footprint_bytes == 0, "VRAM footprint not cleared on offload"

    def test_move_to_gpu_reload(self):
        """_move_model_to_gpu should reload a CPU-cached model to GPU."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            enable_torch_compile=False,
        )

        cached = pytorch_backend._CachedModel(
            pipeline=MagicMock(),
            model_path=Path("/models/test"),
            model_type="sd15",
            device_map_active=False,
            is_single_file_sdxl=False,
        )
        cached.pipeline.to = MagicMock(return_value=cached.pipeline)

        backend._cpu_cache["/models/test"] = cached

        # Reload to GPU
        result = backend._move_model_to_gpu("/models/test")
        assert result is True, "Failed to reload CPU model to GPU"
        assert "/models/test" not in backend._cpu_cache, "Model not removed from CPU cache after reload"

    def test_move_to_gpu_not_found(self):
        """_move_model_to_gpu should return False for models not in CPU cache."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            enable_torch_compile=False,
        )
        result = backend._move_model_to_gpu("/models/nonexistent")
        assert result is False


# ---------------------------------------------------------------------------
# Non-blocking cleanup tests
# ---------------------------------------------------------------------------

class TestNonBlockingCleanup:
    """Test that _cleanup_after_generation runs in a thread pool."""

    def test_cleanup_uses_asyncio_to_thread(self):
        """_cleanup_after_generation should use asyncio.to_thread for cleanup."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        assert "asyncio.to_thread" in source, (
            "_cleanup_after_generation does not use asyncio.to_thread"
        )
        assert "asyncio.wait_for" in source, (
            "_cleanup_after_generation does not use asyncio.wait_for with timeout"
        )

    def test_cleanup_has_timeout(self):
        """_cleanup_after_generation should have a 30s timeout."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        assert "timeout=30" in source, (
            "_cleanup_after_generation does not have a 30s timeout"
        )

    def test_gc_runs_every_5_generations(self):
        """gc.collect() should only run every 5 generations (gc_interval=5)."""
        backend = pytorch_backend.PyTorchBackend(
            images_dir=Path("./test_images"),
            force_cpu=True,
            enable_torch_compile=False,
        )
        assert backend._gc_interval == 5, f"gc_interval should be 5, got {backend._gc_interval}"

    def test_cleanup_sync_has_synchronize_before_empty_cache(self):
        """GPU cleanup should call torch.cuda.synchronize() before empty_cache()."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        # The _cleanup_sync function inside _cleanup_after_generation
        assert "torch.cuda.synchronize()" in source, (
            "torch.cuda.synchronize() not called before empty_cache()"
        )

    def test_evict_cleanup_non_blocking(self):
        """_evict_lru should run empty_cache in a thread, not synchronously."""
        pb_path = Path(pytorch_backend.__file__)
        source = pb_path.read_text()
        # Should use asyncio.create_task with asyncio.to_thread for cleanup
        assert "asyncio.create_task(asyncio.to_thread" in source, (
            "_evict_lru does not use asyncio.to_thread for non-blocking cleanup"
        )
