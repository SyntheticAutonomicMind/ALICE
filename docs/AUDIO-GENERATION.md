# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: Copyright (c) 2026 Andrew Wyatt (Fewtarius)

"""
ALICE Audio Generation

Stable Audio Open 1.0 runs on the same ROCm / CUDA / MPS torch stack as
the image backends.  This document covers what is supported, how to set
it up, and how the VRAM context switching works.

## Supported models

| Model id                 | Source repo                            | Max length | Sample rate | Status      | Engine            |
| ------------------------ | -------------------------------------- | ---------- | ----------- | ----------- | ----------------- |
| `stable-audio-open-1.0`  | `stabilityai/stable-audio-open-1.0`    | 47 s       | 44.1 kHz    | Enabled     | `stable_audio`    |
| `minimax-music-3`        | `MiniMaxAI/MiniMax-Music3`             | 360 s      | 44.1 kHz    | Enabled*    | `minimax_music3`  |

\* MiniMax-Music3 is enabled when both `diffusers` (git main with
PR #14456) and the MiniMax-Music3 model weights are present.  The model
runs on the same ROCm / CUDA / MPS torch stack as the image backends;
no separate CUDA build is required.  Test it with:

```bash
python -c "from diffusers.modular_pipelines.minimax_music3 import MiniMaxMusic3ModularPipeline"
```

### MiniMax-Music3 vs Stable Audio: parameter differences

The two engines accept different request shapes.  `/v1/audio/generations`
hides the difference behind a single endpoint:

| Field           | Stable Audio Open 1.0 | MiniMax-Music3                              |
| --------------- | -------------------- | ------------------------------------------- |
| `prompt`        | Sound description    | Music description (genre, mood, instruments) |
| `lyrics`        | ignored              | Lyrics with `[verse]` / `[chorus]` tags      |
| `seconds`       | integer, 1..47       | integer, 1..360 (clamped to 6 minutes)       |
| `steps`         | diffusion steps      | flow-matching Euler steps per chunk         |
| `cfg_scale`     | used                 | ignored (flow-matching has no CFG)           |

### Hardware notes

- **Strix Halo (gfx1151 / 8060S Graphics):** works out of the box on
  the gfx1151 wheel index.  Generation is slow (~25x realtime for a
  30s clip) because the Qwen3-8B AR stage is the bottleneck.  VRAM
  peaks around 22 GB; you have plenty on the 128 GB unified-memory
  Strix Halo platform.
- **Phoenix (gfx1103):** the trial run on gfx1151 wheels doesn't
  translate; keep using gfx110X-all wheels and `vae_decode_cpu: true`.
  MiniMax-Music3 is unlikely to ever run there (22 GB VRAM > the
  integrated GPU's accessible memory).
- **CUDA / NVIDIA:** both engines work on stock PyTorch wheels.

## Installation

`stable-audio-tools` and `torchaudio` are intentionally unpinned in
`requirements.txt`.  The platform-specific `torch` (ROCm / CUDA / CPU)
must already be installed before the audio backend will import.

```bash
# From the ALICE directory, after the torch install step in README.md:
pip install -r requirements.txt
```

Verify the library is importable:

```bash
python -c "from stable_audio_tools import get_pretrained_model; print('ok')"
```

## ROCm / AMD APU notes

The audio backend follows the same rules the image backends use:

- `device = "cuda" if torch.cuda.is_available() else "cpu"`  (ROCm
  exposes the CUDA API, so this resolves to the AMD GPU).
- `torch.amp.autocast(device_type="cuda", dtype=torch.float16)` for
  the diffusion loop - the modern, device-aware API.  Not the
  deprecated `torch.cuda.amp.autocast`.
- `vae_decode_cpu: true` in `config.audio` mirrors the image-side
  flag and is required on AMD gfx1103 (Phoenix APU) to avoid the GPU
  hang during VAE decode.
- `force_fp32: true` if you also need that for image generation on
  the same hardware.

No library source code is patched at runtime.  If `stable-audio-tools`
upstream adds a hardcoded `.cuda()` call we'll route around it via a
shim in `src/audio_engine.py` rather than monkey-patching the venv.

## VRAM context switching

Audio and image generation share the same GPU.  ALICE handles the
collision two ways:

1. **Audio unloads after each generation** by default
   (`audio.unload_after_generate: true`).  The next image request
   finds an empty GPU and loads the requested model normally.
2. **Image generation evicts audio before loading**.  The image
   backend registers an eviction callback at startup that the audio
   backend fires before acquiring the GPU lock.  So an image request
   that arrives mid-audio will wait for the audio lock to release,
   then the audio model is dropped before the image model loads.

Set `audio.unload_after_generate: false` if you are chaining many
audio requests and want to amortise the model load cost.  Image
generation will still be safe - the eviction callback will drop
the audio model on demand.

## API

### `POST /v1/audio/generations`

OpenAI-compatible shape.  Either `prompt` or `input` may be sent.

```json
{
  "prompt": "A warm acoustic guitar loop, 90 BPM",
  "model": "stable-audio-open-1.0",
  "seconds": 30,
  "steps": 100,
  "cfg_scale": 7.0,
  "seed": 42
}
```

Response:

```json
{
  "url": "/v1/audio/alice_12345678_seed42.wav",
  "model": "stable-audio-open-1.0",
  "duration_seconds": 30.0,
  "sample_rate": 44100,
  "seed": 42,
  "steps": 100,
  "cfg_scale": 7.0,
  "prompt": "A warm acoustic guitar loop, 90 BPM",
  "generation_time_seconds": 42.1,
  "size_bytes": 2649600
}
```

### `GET /v1/audio/models`

Lists the catalog.  Disabled models are filtered out so the response
matches what the server can actually run.

### `GET /v1/audio/stats`

Diagnostic snapshot: backend name, availability, loaded model, etc.

### `GET /v1/audio/{filename}`

Serves a generated WAV file.  Path traversal is rejected.

## Configuration

```yaml
audio:
  enabled: true
  default_model: stable-audio-open-1.0
  default_seconds: 30
  default_steps: 100
  default_cfg_scale: 7.0
  max_concurrent: 1
  unload_after_generate: true
  request_timeout_seconds: 300
  force_fp32: false
  vae_decode_cpu: false
```

Audio output path is `storage.audio_directory` (defaults to
`./audio`, the running config points it at
`/home/deck/ALICE.data/audio`).

## License

Stable Audio Open 1.0 is governed by the Stable Audio Community
License.  Commercial use requires a separate license from
https://stability.ai/license.  ALICE does not bundle the model
weights; you download them from HuggingFace on first use.
