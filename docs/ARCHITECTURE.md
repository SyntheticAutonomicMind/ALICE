<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# ALICE Architecture

**Version:** 1.1  
**Purpose:** Remote Stable Diffusion service for SAM integration

---

## Overview

ALICE is a standalone Python service that provides OpenAI-compatible API endpoints for Stable Diffusion image generation. It enables SAM clients to offload image generation to remote GPU servers.

**Key Features:**
- OpenAI-compatible REST API (`POST /v1/chat/completions`)
- Web-based management interface
- Model download manager (CivitAI & HuggingFace)
- Linux daemon deployment
- Multi-user queue management
- Model hot-swapping
- Image generation via `diffusers` library
- Session-based authentication

---

## Architecture Components

### 1. REST API Server (FastAPI)

**Primary Endpoint:** `POST /v1/chat/completions`

**Request Format (OpenAI-compatible):**
```json
{
  "model": "sd/stable-diffusion-v1-5",
  "messages": [
    {
      "role": "user",
      "content": "a serene mountain landscape at sunset"
    }
  ],
  "temperature": 0.7,
  "sam_config": {
    "negative_prompt": "blurry, low quality",
    "steps": 25,
    "guidance_scale": 7.5,
    "width": 512,
    "height": 512,
    "seed": 42,
    "scheduler": "dpm++_sde_karras"
  }
}
```

**Response Format:**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "sd/stable-diffusion-v1-5",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Image generated successfully",
      "image_urls": ["http://server:8080/images/abc123.png"],
      "metadata": {
        "prompt": "a serene mountain landscape at sunset",
        "negative_prompt": "blurry, low quality",
        "steps": 25,
        "guidance_scale": 7.5,
        "seed": 42,
        "model": "stable-diffusion-v1-5",
        "scheduler": "dpm++_sde_karras"
      }
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0
  }
}
```

**Model Discovery:** `GET /v1/models`

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "sd/stable-diffusion-v1-5",
      "object": "model",
      "created": 1234567890,
      "owned_by": "alice"
    },
    {
      "id": "sd/stable-diffusion-xl-base",
      "object": "model",
      "created": 1234567890,
      "owned_by": "alice"
    }
  ]
}
```

### 2. Image Generation Engine

**Based on:** SAM's `generate_image_diffusers.py`

**Pipeline:**
1. Parse request → extract prompt + parameters
2. Load/switch model (if needed)
3. Configure scheduler
4. Generate image(s)
5. Save to static directory
6. Return URL(s) in response

**Model Management:**
- Models stored in `/var/lib/alice/models/`
- Auto-detection of `.safetensors` and diffusers directories
- Hot-swapping without server restart
- Model registry file for metadata

**Supported Features:**
- Text-to-image
- Image-to-image (future)
- Multiple schedulers (DPM++, Euler, DDIM, etc.)
- Seed control for reproducibility
- Dynamic resolution (model-dependent)

### 3. Web Management Interface

**Technology:** HTML + JavaScript (vanilla or lightweight framework)

**Features:**

**Dashboard:**
- Active generation queue
- GPU utilization
- Model status (loaded/unloaded)
- Recent generations gallery

**Model Management:**
- List installed models
- Download from HuggingFace/CivitAI
- Delete models
- Set default model

**Model Download Page (Admin only):**
- Browse popular CivitAI models (auto-loads on page open)
- Local search filtering (instant, searches name/creator/tags)
- HuggingFace model search via API
- Real-time download progress
- Type filtering (Checkpoint, LoRA, etc.)

**Generation Queue:**
- View pending requests
- Cancel requests
- Priority management
- User/client identification

**Settings:**
- API key management
- Port configuration
- Storage limits
- Default generation parameters
- Session timeout configuration

**API Documentation:**
- Interactive API explorer
- Example requests
- Model capabilities table

### 4. Model Download Manager

**Supported Sources:**
- CivitAI (Checkpoints, LoRAs, TextualInversion, etc.)
- HuggingFace Hub (diffusers models)

**CivitAI Integration:**

The CivitAI API has specific quirks that require special handling:
- **Browse Mode** (empty query): Server-side type filtering works correctly, pagination available
- **Search Mode** (with query): Type filtering is unreliable, must filter client-side

```python
# Browse popular checkpoints (works correctly)
POST /v1/models/search/civitai
{"query": "", "types": ["Checkpoint"], "page": 1, "limit": 100}

# Search with query (client-side filtering)
POST /v1/models/search/civitai
{"query": "stable diffusion", "types": ["Checkpoint"]}
```

**HuggingFace Integration:**

Uses `huggingface_hub.snapshot_download()` for reliable model downloads:
- Handles authentication automatically
- Supports resume on interruption
- Downloads only required files (filters out docs, READMEs)
- No git-lfs dependency required

```python
POST /v1/models/download/huggingface
{"repoId": "stabilityai/stable-diffusion-xl-base-1.0"}
```

**Download Queue:**
- Background task processing with asyncio
- Progress tracking via `/v1/models/download/status`
- Cancelable downloads

### 5. Linux Daemon Deployment

**Service Manager:** systemd

**Service File:** `/etc/systemd/system/alice.service`

```ini
[Unit]
Description=ALICE Stable Diffusion Service
After=network.target

[Service]
Type=simple
User=alice
Group=alice
WorkingDirectory=/opt/alice
Environment="PATH=/opt/alice/venv/bin"
ExecStart=/opt/alice/venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10

[Install]
WantedBy=multi-tier.target
```

**Deployment Structure:**
```
/opt/alice/
├── venv/                      # Python virtual environment
├── src/                       # Application source
│   ├── main.py               # FastAPI app
│   ├── generator.py          # SD generation logic
│   ├── models.py             # Model management
│   └── queue.py              # Request queue
├── web/                       # Static web UI
│   ├── index.html
│   ├── models.html
│   └── queue.html
├── config.yaml               # Service configuration
└── requirements.txt          # Python dependencies

/var/lib/alice/
├── models/                   # SD model files
├── images/                   # Generated images
└── cache/                    # Temporary files

/var/log/alice/
└── alice.log               # Service logs
```

---

## SAM Integration

### Provider Implementation

**File:** `Sources/APIFramework/RemoteStableDiffusionProvider.swift`

**Pattern:** Follows `CustomProvider` architecture

**Key Differences from Local SD:**
- Network calls instead of local Python execution
- URL-based image retrieval
- No model loading in SAM
- Supports multiple simultaneous clients

**Integration Points:**

1. **EndpointManager:**
   - Add `remoteStableDiffusion` to `ProviderType` enum
   - Register provider in `setupProviders()`

2. **Provider Configuration:**
   ```swift
   ProviderConfiguration(
       providerId: "remote-sd",
       providerType: .remoteStableDiffusion,
       isEnabled: true,
       baseURL: "http://gpu-server:8080",
       models: []  // Fetched from /v1/models
   )
   ```

3. **Request Flow:**
   ```
   SAM generate_image tool
   → RemoteStableDiffusionProvider
   → HTTP POST to remote server
   → Download image from URL
   → Save to local cache
   → Return local path to tool
   ```

4. **Model Picker:**
   - Prefix: `remote-sd/` (e.g., `remote-sd/stable-diffusion-v1-5`)
   - Fetched dynamically from `/v1/models`
   - Updated on provider reload

---

## API Specification

### Endpoints

**Chat Completions (Image Generation):**
```
POST /v1/chat/completions
Content-Type: application/json
Authorization: Bearer <api-key>

Request: OpenAIChatRequest (see format above)
Response: OpenAIChatResponse with image_urls
```

**Model List:**
```
GET /v1/models
Authorization: Bearer <api-key>

Response: OpenAIModelsResponse
```

**Health Check:**
```
GET /health

Response: { "status": "ok", "gpu_available": true, "models_loaded": 1 }
```

**Image Retrieval:**
```
GET /images/<image_id>.png

Response: PNG image binary data
```

### Error Handling

**Error Response Format:**
```json
{
  "error": {
    "message": "Model not found: sd/invalid-model",
    "type": "invalid_request_error",
    "code": "model_not_found"
  }
}
```

**Error Codes:**
- `model_not_found` - Requested model doesn't exist
- `invalid_parameters` - Invalid generation parameters
- `gpu_out_of_memory` - Insufficient GPU memory
- `generation_failed` - Image generation error
- `rate_limit_exceeded` - Too many requests

---

## Security

**Authentication:**
- API key in `Authorization: Bearer <key>` header
- Configurable in `config.yaml`
- Optional (can run without auth on trusted networks)

**Network Security:**
- HTTPS support (reverse proxy recommended)
- CORS configuration
- Rate limiting
- IP whitelist (optional)

**File System:**
- Sandboxed model directory
- Image cleanup policy (configurable retention)
- Disk space monitoring

---

## Performance Considerations

**GPU Management:**
- Single model loaded at a time (avoid OOM)
- Model unloading after idle timeout
- GPU memory monitoring

**Queue Management:**
- FIFO queue with priority support
- Concurrent generation limit: 1 (sequential)
- Request timeout: configurable (default 300s)

**Image Storage:**
- Automatic cleanup (>7 days old)
- Configurable max storage
- Compression for archived images

**Caching:**
- Model metadata cache
- Scheduler configurations
- No image result caching (each generation unique)

---

## Monitoring & Logging

**Logs:**
- Request/response logging
- Error tracking
- Performance metrics (generation time, queue depth)
- GPU utilization

**Metrics Endpoint:**
```
GET /metrics

Response:
{
  "queue_depth": 3,
  "gpu_utilization": 0.85,
  "gpu_memory_used": "10.2 GB",
  "gpu_memory_total": "24 GB",
  "models_loaded": 1,
  "total_generations": 1523,
  "avg_generation_time": 12.4
}
```

---

## Deployment Guide

### Prerequisites

**System Requirements:**
- Linux (Ubuntu 22.04+ recommended)
- NVIDIA GPU with CUDA support
- 24GB+ GPU memory (for SDXL)
- 50GB+ disk space
- Python 3.10+

**Dependencies:**
```bash
# System packages
apt-get install python3 python3-venv python3-pip nginx

# CUDA toolkit (for GPU support)
# Follow NVIDIA official instructions
```

### Installation

1. **Create service user:**
```bash
sudo useradd -r -s /bin/false alice
sudo mkdir -p /opt/alice /var/lib/alice /var/log/alice
sudo chown alice:alice /var/lib/alice /var/log/alice
```

2. **Install application:**
```bash
cd /opt/alice
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Configure service:**
```bash
sudo cp alice.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable alice
sudo systemctl start alice
```

4. **Configure nginx (reverse proxy):**
```nginx
server {
    listen 80;
    server_name alice.example.com;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    location /images/ {
        alias /var/lib/alice/images/;
    }
}
```

### Configuration

**File:** `/opt/alice/config.yaml`

```yaml
server:
  host: 0.0.0.0
  port: 8080
  api_key: your-secret-key-here  # Optional

models:
  directory: /var/lib/alice/models
  auto_unload_timeout: 300  # Seconds
  default_model: stable-diffusion-v1-5

generation:
  default_steps: 25
  default_guidance_scale: 7.5
  default_scheduler: dpm++_sde_karras
  max_concurrent: 1
  request_timeout: 300

storage:
  images_directory: /var/lib/alice/images
  max_storage_gb: 100
  retention_days: 7

logging:
  level: INFO
  file: /var/log/alice/alice.log
  max_size_mb: 100
  backup_count: 5
```

---

## Future Enhancements

**Planned Features:**
- Image-to-image support
- ControlNet integration
- LoRA loading
- Upscaling (ESRGAN, Real-ESRGAN)
- Video generation (AnimateDiff)
- Multi-GPU support
- Distributed queue (Redis)
- WebSocket streaming (progress updates)

**SAM Integration Improvements:**
- Streaming progress in chat
- Batch generation support
- Model recommendations based on prompt
- Automatic parameter tuning

---

## Troubleshooting

**Common Issues:**

1. **GPU Out of Memory:**
   - Reduce image resolution
   - Use model with lower memory requirements
   - Enable CPU offloading (slower)

2. **Slow Generation:**
   - Check GPU utilization
   - Verify CUDA installation
   - Consider using faster scheduler (Euler)

3. **Model Loading Fails:**
   - Check model file integrity
   - Verify disk space
   - Check file permissions

4. **Network Issues:**
   - Verify firewall rules
   - Check nginx configuration
   - Test with curl: `curl http://localhost:8080/health`

---

## References

- SAM generate_image_diffusers.py implementation
- OpenAI Chat Completions API spec
- Hugging Face Diffusers documentation
- FastAPI documentation
- systemd service configuration
