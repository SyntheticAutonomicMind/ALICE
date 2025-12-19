<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<!-- SPDX-FileCopyrightText: Copyright (c) 2025 Andrew Wyatt (Fewtarius) -->

# ALICE Implementation Guide

**Version:** 1.1  
**Purpose:** Step-by-step implementation guide for ALICE service

---

## Overview

This guide walks through implementing ALICE from scratch. Follow the phases in order for a working remote Stable Diffusion service.

**Total Estimated Time:** 12-16 hours  
**Skill Requirements:** Python, FastAPI, Linux, systemd, basic Docker knowledge

---

## Phase 1: Core API Server (4-6 hours)

### Step 1.1: Project Setup

**Create virtual environment:**
```bash
cd /opt/alice
python3 -m venv venv
source venv/bin/activate
```

**Create requirements.txt:**
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
torch==2.1.0
torchvision==0.16.0
diffusers==0.25.0
transformers==4.36.0
accelerate==0.25.0
safetensors==0.4.1
pillow==10.1.0
pyyaml==6.0.1
aiofiles==23.2.1
python-multipart==0.0.6
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### Step 1.2: Configuration Module

**File:** `src/config.py`

```python
import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: Optional[str] = None

class ModelsConfig(BaseModel):
    directory: Path = Path("/var/lib/alice/models")
    auto_unload_timeout: int = 300
    default_model: str = "stable-diffusion-v1-5"

class GenerationConfig(BaseModel):
    default_steps: int = 25
    default_guidance_scale: float = 7.5
    default_scheduler: str = "dpm++_sde_karras"
    max_concurrent: int = 1
    request_timeout: int = 300

class StorageConfig(BaseModel):
    images_directory: Path = Path("/var/lib/alice/images")
    max_storage_gb: int = 100
    retention_days: int = 7

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Path = Path("/var/log/alice/alice.log")
    max_size_mb: int = 100
    backup_count: int = 5

class Config(BaseModel):
    server: ServerConfig
    models: ModelsConfig
    generation: GenerationConfig
    storage: StorageConfig
    logging: LoggingConfig

def load_config(path: str = "/opt/alice/config.yaml") -> Config:
    """Load configuration from YAML file."""
    if Path(path).exists():
        with open(path) as f:
            data = yaml.safe_load(f)
            return Config(**data)
    else:
        # Return default config
        return Config(
            server=ServerConfig(),
            models=ModelsConfig(),
            generation=GenerationConfig(),
            storage=StorageConfig(),
            logging=LoggingConfig()
        )

# Global config instance
config = load_config()
```

### Step 1.3: Model Registry

**File:** `src/model_registry.py`

```python
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelInfo:
    id: str
    name: str
    path: Path
    type: str  # "sd15", "sdxl", "custom"
    created: int
    size_mb: int
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.path),
            "type": self.type,
            "created": self.created,
            "size_mb": self.size_mb
        }

class ModelRegistry:
    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / ".registry.json"
        self.models: Dict[str, ModelInfo] = {}
        self.load_registry()
    
    def scan_models(self) -> List[ModelInfo]:
        """Scan models directory and update registry."""
        found_models = []
        
        # Scan for .safetensors files
        for safetensors_file in self.models_dir.rglob("*.safetensors"):
            if safetensors_file.is_file():
                model_id = f"sd/{safetensors_file.stem}"
                model_info = ModelInfo(
                    id=model_id,
                    name=safetensors_file.stem,
                    path=safetensors_file,
                    type="sd15",  # TODO: Auto-detect type
                    created=int(safetensors_file.stat().st_ctime),
                    size_mb=int(safetensors_file.stat().st_size / 1024 / 1024)
                )
                found_models.append(model_info)
                self.models[model_id] = model_info
        
        # Scan for diffusers directories
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and (model_dir / "model_index.json").exists():
                model_id = f"sd/{model_dir.name}"
                model_info = ModelInfo(
                    id=model_id,
                    name=model_dir.name,
                    path=model_dir,
                    type="sdxl" if "xl" in model_dir.name.lower() else "sd15",
                    created=int(model_dir.stat().st_ctime),
                    size_mb=sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) // 1024 // 1024
                )
                found_models.append(model_info)
                self.models[model_id] = model_info
        
        self.save_registry()
        return found_models
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info by ID."""
        return self.models.get(model_id)
    
    def list_models(self) -> List[ModelInfo]:
        """List all registered models."""
        return list(self.models.values())
    
    def load_registry(self):
        """Load registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file) as f:
                data = json.load(f)
                for model_data in data.get("models", []):
                    model_info = ModelInfo(
                        id=model_data["id"],
                        name=model_data["name"],
                        path=Path(model_data["path"]),
                        type=model_data["type"],
                        created=model_data["created"],
                        size_mb=model_data["size_mb"]
                    )
                    self.models[model_info.id] = model_info
        else:
            # First run - scan models
            self.scan_models()
    
    def save_registry(self):
        """Save registry to file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated": int(datetime.now().timestamp()),
            "models": [model.to_dict() for model in self.models.values()]
        }
        with open(self.registry_file, "w") as f:
            json.dump(data, f, indent=2)
```

### Step 1.4: Image Generator

**File:** `src/generator.py`

**Reference:** Use `docs/sam-reference/generate_image_diffusers.py` as base

**Key modifications:**
1. Remove argparse CLI interface
2. Create `GeneratorService` class
3. Add model caching (keep loaded models in memory)
4. Return image path instead of saving to user-specified path

**Pseudo-code structure:**
```python
class GeneratorService:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.current_model = None
        self.current_pipeline = None
    
    async def generate_image(
        self,
        model_path: Path,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 25,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: Optional[int] = None,
        scheduler: str = "dpm++_sde_karras"
    ) -> Path:
        # Load model if different from current
        if self.current_model != model_path:
            await self.load_model(model_path)
        
        # Generate image using pipeline
        # Save to storage directory
        # Return path
        
        pass
    
    async def load_model(self, model_path: Path):
        # Based on generate_image_diffusers.py logic
        pass
```

**Copy and adapt from generate_image_diffusers.py:**
- Pipeline detection logic
- Scheduler configuration
- Device management (CUDA/CPU)
- Memory optimization

### Step 1.5: Main FastAPI App

**File:** `src/main.py`

```python
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import time
from pathlib import Path
import logging

from .config import config
from .model_registry import ModelRegistry
from .generator import GeneratorService

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.logging.level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(title="ALICE", version="1.0.0")

# Initialize services
model_registry = ModelRegistry(config.models.directory)
generator = GeneratorService(config.generation)

# Mount static files for images
app.mount("/images", StaticFiles(directory=str(config.storage.images_directory)), name="images")

# ============================================================================
# OPENAI-COMPATIBLE MODELS
# ============================================================================

class ChatMessage(BaseModel):
    role: str
    content: str

class SamConfig(BaseModel):
    negative_prompt: Optional[str] = ""
    steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    width: Optional[int] = 512
    height: Optional[int] = 512
    seed: Optional[int] = None
    scheduler: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    sam_config: Optional[SamConfig] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class Model(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str

class ModelsResponse(BaseModel):
    object: str
    data: List[Model]

# ============================================================================
# AUTHENTICATION
# ============================================================================

async def verify_api_key(authorization: Optional[str] = Header(None)):
    """Verify API key if configured."""
    if config.server.api_key:
        if not authorization:
            raise HTTPException(status_code=401, detail="Missing API key")
        
        # Extract Bearer token
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        
        token = authorization[7:]
        if token != config.server.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
    
    return True

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "gpu_available": True,  # TODO: Check actual GPU
        "models_loaded": 1 if generator.current_model else 0
    }

@app.get("/v1/models", response_model=ModelsResponse)
async def list_models(authenticated: bool = Depends(verify_api_key)):
    """List available models."""
    models = model_registry.list_models()
    
    return ModelsResponse(
        object="list",
        data=[
            Model(
                id=model.id,
                object="model",
                created=model.created,
                owned_by="alice"
            )
            for model in models
        ]
    )

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    authenticated: bool = Depends(verify_api_key)
):
    """
    Generate image via OpenAI-compatible chat completions endpoint.
    
    The prompt is extracted from the last user message.
    Generation parameters come from sam_config.
    """
    
    # Extract prompt from messages
    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")
    
    prompt = user_messages[-1].content
    
    # Get model info
    model_info = model_registry.get_model(request.model)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model}")
    
    # Extract generation parameters
    sam_config = request.sam_config or SamConfig()
    
    steps = sam_config.steps or config.generation.default_steps
    guidance_scale = sam_config.guidance_scale or config.generation.default_guidance_scale
    scheduler = sam_config.scheduler or config.generation.default_scheduler
    
    # Generate image
    try:
        logger.info(f"Generating image: {prompt[:50]}...")
        
        image_path = await generator.generate_image(
            model_path=model_info.path,
            prompt=prompt,
            negative_prompt=sam_config.negative_prompt or "",
            steps=steps,
            guidance_scale=guidance_scale,
            width=sam_config.width or 512,
            height=sam_config.height or 512,
            seed=sam_config.seed,
            scheduler=scheduler
        )
        
        # Generate response
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        image_url = f"http://{config.server.host}:{config.server.port}/images/{image_path.name}"
        
        response_content = f"Image generated successfully. URL: {image_url}"
        
        return ChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=response_content
                    ),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0
            )
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Metrics endpoint for monitoring."""
    # TODO: Implement actual metrics
    return {
        "queue_depth": 0,
        "gpu_utilization": 0.0,
        "gpu_memory_used": "0 GB",
        "gpu_memory_total": "24 GB",
        "models_loaded": 1 if generator.current_model else 0,
        "total_generations": 0,
        "avg_generation_time": 0.0
    }

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize services on startup."""
    logger.info("ALICE starting up...")
    
    # Scan for models
    models = model_registry.scan_models()
    logger.info(f"Found {len(models)} models")
    
    # Create storage directory
    config.storage.images_directory.mkdir(parents=True, exist_ok=True)
    
    logger.info("ALICE ready")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False
    )
```

**Testing Phase 1:**
```bash
# Start server
python -m src.main

# Test health
curl http://localhost:8080/health

# Test models list
curl http://localhost:8080/v1/models

# Test generation (requires model installed)
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sd/stable-diffusion-v1-5",
    "messages": [{"role": "user", "content": "a cat"}],
    "sam_config": {"steps": 20}
  }'
```

---

## Phase 2: Web Management Interface (3-4 hours)

### Step 2.1: Static Web UI Structure

**Directory:** `web/`

**Files:**
- `web/index.html` - Dashboard
- `web/models.html` - Model management
- `web/queue.html` - Generation queue
- `web/settings.html` - Configuration
- `web/style.css` - Shared styles
- `web/app.js` - Shared JavaScript

### Step 2.2: Dashboard (index.html)

**Features:**
- Server status
- GPU metrics
- Recent generations gallery
- Quick generation form

**Implementation:**
```html
<!DOCTYPE html>
<html>
<head>
    <title>ALICE Dashboard</title>
    <link rel="stylesheet" href="/web/style.css">
</head>
<body>
    <nav>
        <a href="/web/">Dashboard</a>
        <a href="/web/models.html">Models</a>
        <a href="/web/queue.html">Queue</a>
        <a href="/web/settings.html">Settings</a>
    </nav>
    
    <main>
        <h1>ALICE Dashboard</h1>
        
        <section class="metrics">
            <div class="metric">
                <h3>Status</h3>
                <p id="status">Loading...</p>
            </div>
            <div class="metric">
                <h3>GPU</h3>
                <p id="gpu">Loading...</p>
            </div>
            <div class="metric">
                <h3>Models</h3>
                <p id="models-loaded">Loading...</p>
            </div>
        </section>
        
        <section class="quick-gen">
            <h2>Quick Generation</h2>
            <form id="gen-form">
                <select id="model-select"></select>
                <textarea id="prompt" placeholder="Prompt..."></textarea>
                <button type="submit">Generate</button>
            </form>
        </section>
        
        <section class="recent">
            <h2>Recent Generations</h2>
            <div id="gallery"></div>
        </section>
    </main>
    
    <script src="/web/app.js"></script>
    <script>
        // Load metrics
        async function loadMetrics() {
            const res = await fetch('/health');
            const data = await res.json();
            document.getElementById('status').textContent = data.status;
            document.getElementById('models-loaded').textContent = data.models_loaded;
        }
        
        // Load models
        async function loadModels() {
            const res = await fetch('/v1/models');
            const data = await res.json();
            const select = document.getElementById('model-select');
            data.data.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.id;
                select.appendChild(option);
            });
        }
        
        loadMetrics();
        loadModels();
        setInterval(loadMetrics, 5000);
    </script>
</body>
</html>
```

### Step 2.3: Mount Web UI in FastAPI

**Update src/main.py:**
```python
# Add to imports
from fastapi.staticfiles import StaticFiles

# Mount web UI
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# Redirect root to web UI
@app.get("/")
async def root():
    return {"message": "ALICE running. Visit /web for UI."}
```

---

## Phase 3: Model Download Manager (2-3 hours)

### Step 3.1: CivitAI API Integration

**Key insight:** The CivitAI API behaves differently based on whether a query is provided:

```python
# Browse mode (no query) - server-side filtering works
params = {
    "types": "Checkpoint",  # Works correctly
    "sort": "Highest Rated",
    "limit": 100,
    "page": 1
}

# Search mode (with query) - must filter client-side
params = {
    "query": "stable diffusion",
    "sort": "Highest Rated",
    "limit": 100
    # types filter is unreliable here
}
```

**Implementation pattern:**
```python
async def search_civitai(self, query: str, types: list, ...):
    if query and query.strip():
        # Search mode - client-side filtering
        params["query"] = query
        use_client_side_filter = True
    else:
        # Browse mode - server-side filtering
        params["page"] = page
        if types:
            params["types"] = ",".join(types)
        use_client_side_filter = False
```

### Step 3.2: HuggingFace Integration

Use `huggingface_hub` instead of git clone for reliability:

```python
from huggingface_hub import snapshot_download

async def download_huggingface(self, repo_id: str, ...):
    local_path = await asyncio.to_thread(
        snapshot_download,
        repo_id=repo_id,
        local_dir=destination,
        ignore_patterns=["*.md", "*.txt", ".git*"],
        token=self.huggingface_token
    )
    return local_path
```

**Benefits over git clone:**
- No git-lfs dependency
- Automatic resume on interruption
- Better progress tracking
- Handles authentication cleanly

### Step 3.3: Download Queue

```python
class DownloadTask:
    id: str
    name: str
    source: str  # "civitai" or "huggingface"
    status: str  # "queued", "downloading", "completed", "failed"
    progress: float
    error: Optional[str]

# Background processing
async def process_download_queue():
    while True:
        if download_queue:
            task = download_queue.pop(0)
            await process_task(task)
        await asyncio.sleep(1)
```

---

## Phase 4: Linux Daemon Deployment (2-3 hours)

### Step 4.1: System User & Directories

```bash
# Create service user
sudo useradd -r -s /bin/false alice

# Create directories
sudo mkdir -p /opt/alice
sudo mkdir -p /var/lib/alice/{models,images}
sudo mkdir -p /var/log/alice

# Set permissions
sudo chown -R alice:alice /var/lib/alice /var/log/alice
```

### Step 3.2: systemd Service File

**File:** `/etc/systemd/system/alice.service`

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
Environment="PYTHONUNBUFFERED=1"
ExecStart=/opt/alice/venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8080
Restart=always
RestartSec=10
StandardOutput=append:/var/log/alice/alice.log
StandardError=append:/var/log/alice/alice.log

[Install]
WantedBy=multi-user.target
```

### Step 3.3: Deployment Script

**File:** `deploy.sh`

```bash
#!/bin/bash
set -e

echo "Deploying ALICE..."

# Copy files
sudo cp -r src web requirements.txt config.yaml /opt/alice/

# Install dependencies
cd /opt/alice
sudo -u alice python3 -m venv venv
sudo -u alice /opt/alice/venv/bin/pip install -r requirements.txt

# Install systemd service
sudo cp alice.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable alice
sudo systemctl restart alice

echo "Deployment complete!"
echo "Service status:"
sudo systemctl status alice
```

---

## Phase 5: SAM Provider Integration (3-4 hours)

### Step 4.1: Add Provider Type

**File:** `Sources/ConfigurationSystem/EndpointConfigurationModels.swift`

**Add to ProviderType enum:**
```swift
case remoteStableDiffusion
```

**Add to computed properties:**
```swift
public var defaultIdentifier: String {
    switch self {
    // ... existing cases
    case .remoteStableDiffusion: return "remote-sd"
    }
}

public var requiresApiKey: Bool {
    switch self {
    // ... existing cases
    case .remoteStableDiffusion: return false  // Optional
    }
}

public var defaultBaseURL: String? {
    switch self {
    // ... existing cases
    case .remoteStableDiffusion: return "http://localhost:8080"
    }
}
```

### Step 4.2: Create Provider

**File:** `Sources/APIFramework/RemoteStableDiffusionProvider.swift`

**Reference:** Copy structure from `ExtendedProviders.swift` CustomProvider

**Key implementation:**
```swift
import Foundation
import ConfigurationSystem
import Logging

@MainActor
public class RemoteStableDiffusionProvider: AIProvider {
    public let identifier: String
    public let config: ProviderConfiguration
    private let logger = Logger(label: "com.sam.RemoteStableDiffusionProvider")
    
    public init(config: ProviderConfiguration) {
        self.identifier = config.providerId
        self.config = config
        logger.info("Remote SD Provider initialized: \\(config.baseURL ?? "no URL")")
    }
    
    public func processChatCompletion(_ request: OpenAIChatRequest) async throws -> ServerOpenAIChatResponse {
        // 1. Make HTTP POST to remote server
        // 2. Parse response with image URL
        // 3. Download image from URL
        // 4. Save to SAM cache
        // 5. Return response with local path
        
        guard let baseURL = config.baseURL else {
            throw ProviderError.invalidConfiguration("Remote SD base URL not configured")
        }
        
        let url = URL(string: "\\(baseURL)/v1/chat/completions")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        // Add API key if configured
        if let apiKey = config.apiKey {
            urlRequest.setValue("Bearer \\(apiKey)", forHTTPHeaderField: "Authorization")
        }
        
        // Encode request
        let encoder = JSONEncoder()
        urlRequest.httpBody = try encoder.encode(request)
        
        // Make request
        let (data, response) = try await URLSession.shared.data(for: urlRequest)
        
        guard let httpResponse = response as? HTTPURLResponse,
              200...299 ~= httpResponse.statusCode else {
            throw ProviderError.networkError("Remote SD server error")
        }
        
        // Decode response
        let decoder = JSONDecoder()
        let sdResponse = try decoder.decode(ServerOpenAIChatResponse.self, from: data)
        
        // Download image if URL in response
        // TODO: Parse image URL from response content
        // TODO: Download and save to cache
        
        return sdResponse
    }
    
    public func processStreamingChatCompletion(_ request: OpenAIChatRequest) async throws -> AsyncThrowingStream<ServerOpenAIChatStreamChunk, Error> {
        // For images, streaming doesn't make sense - just return final result
        let response = try await processChatCompletion(request)
        
        return AsyncThrowingStream { continuation in
            Task {
                // Convert response to stream chunks
                // ...
                continuation.finish()
            }
        }
    }
    
    public func getAvailableModels() async throws -> ServerOpenAIModelsResponse {
        guard let baseURL = config.baseURL else {
            throw ProviderError.invalidConfiguration("Remote SD base URL not configured")
        }
        
        let url = URL(string: "\\(baseURL)/v1/models")!
        let (data, _) = try await URLSession.shared.data(from: url)
        
        let decoder = JSONDecoder()
        return try decoder.decode(ServerOpenAIModelsResponse.self, from: data)
    }
    
    // Remaining AIProvider protocol methods...
}
```

### Step 4.3: Register Provider

**File:** `Sources/APIFramework/EndpointManager.swift`

**Add to createProvider() method:**
```swift
case .remoteStableDiffusion:
    return RemoteStableDiffusionProvider(config: config)
```

---

## Phase 6: Testing & Validation (1-2 hours)

### Test Checklist

**Python Service:**
- [ ] Health endpoint returns 200
- [ ] Models endpoint lists installed models
- [ ] Chat completions generates image
- [ ] Image URL is accessible
- [ ] Web UI loads correctly
- [ ] Service survives restart
- [ ] Logs are written correctly

**SAM Integration:**
- [ ] Provider shows in preferences
- [ ] Models populate in picker
- [ ] generate_image tool works
- [ ] Image displays in chat
- [ ] Error handling works

**End-to-End:**
- [ ] SAM → Remote SD → Image displayed
- [ ] Multiple requests queue properly
- [ ] GPU memory doesn't leak
- [ ] Service stays responsive

---

## Deployment Checklist

- [ ] Python service installed at /opt/alice
- [ ] systemd service enabled and running
- [ ] Firewall allows port 8080 (or configured port)
- [ ] Models downloaded to /var/lib/alice/models
- [ ] Logs rotating properly
- [ ] Web UI accessible from LAN
- [ ] SAM configured with correct server URL
- [ ] API key configured (if using auth)

---

## Troubleshooting

**Service won't start:**
```bash
sudo journalctl -u alice -n 50
```

**GPU not detected:**
```bash
nvidia-smi
```

**Model loading fails:**
- Check disk space: `df -h`
- Check permissions: `ls -la /var/lib/alice/models`
- Check logs: `tail -f /var/log/alice/alice.log`

**SAM can't connect:**
- Test manually: `curl http://server:8080/health`
- Check firewall: `sudo ufw status`
- Verify SAM config has correct URL

---

## Next Steps

After basic implementation works:
1. Add image-to-image support
2. Implement LoRA loading
3. Add queue management UI
4. Implement metrics collection
5. Add authentication/authorization
6. Setup reverse proxy (nginx)
7. Add HTTPS support
8. Implement auto-scaling (multiple workers)
