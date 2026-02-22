# ALICE API Usage Guide

## Chat Completions Endpoint

**Endpoint**: `POST /v1/chat/completions`

### Headers

```
Content-Type: application/json
Authorization: Bearer <your-api-key>  (if authentication is enabled)
```

### Request Format

```json
{
  "model": "sd/stable-diffusion-v1-5",
  "messages": [
    {
      "role": "user",
      "content": "a cat sitting on a mat"
    }
  ],
  "samConfig": {
    "steps": 30,
    "guidance_scale": 10.0,
    "width": 512,
    "height": 512,
    "negative_prompt": "ugly, blurry",
    "seed": 12345,
    "scheduler": "euler_ancestral",
    "num_images": 1
  }
}
```

### Image-to-Image (img2img) Request

For image editing or style transfer, provide input images via base64 or URL:

```json
{
  "model": "sd/qwen-image-edit-2511-Q2_K",
  "messages": [
    {
      "role": "user",
      "content": "Transform it into Pixar-inspired 3D"
    }
  ],
  "samConfig": {
    "steps": 30,
    "guidance_scale": 2.5,
    "width": 1024,
    "height": 1024,
    "scheduler": "euler",
    "strength": 0.75,
    "input_images": ["<base64-encoded-image-data>"]
  }
}
```

Or provide input images via URL:

```json
{
  "model": "sd/qwen-image-edit-2511-Q2_K",
  "messages": [
    {
      "role": "user",
      "content": "Age the person to 50 years old"
    }
  ],
  "samConfig": {
    "guidance_scale": 2.5,
    "scheduler": "euler",
    "strength": 0.75,
    "input_image_urls": ["https://example.com/photo.jpg"]
  }
}
```

### SAM Config Fields

All fields are **optional** - server uses `config.yaml` defaults if omitted.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `steps` | int | 1-150 | Inference steps (NOT `num_inference_steps`) |
| `guidance_scale` | float | 0.0-20.0 | CFG scale (0.0 for SD Turbo models, 2.5 for Qwen) |
| `width` | int | 64-2048 | Image width (auto-rounded to multiple of 8) |
| `height` | int | 64-2048 | Image height (auto-rounded to multiple of 8) |
| `negative_prompt` | string | - | Negative prompt |
| `seed` | int | - | Random seed for reproducibility |
| `scheduler` | string | - | Scheduler name (see below) |
| `num_images` | int | 1-100 | Number of images to generate |
| `lora_paths` | array[string] | - | List of LoRA IDs to apply |
| `lora_scales` | array[float] | 0.0-1.0 | LoRA weights (same length as `lora_paths`) |
| `strength` | float | 0.0-1.0 | Denoising strength for img2img (0=no change, 1=full denoise) |
| `input_images` | array[string] | - | Base64-encoded input images for img2img |
| `input_image_urls` | array[string] | - | URLs of input images for img2img |

**⚠️ IMPORTANT**: Field names are **case-sensitive**:
- Use `steps`, NOT `num_inference_steps`
- Use `guidance_scale`, NOT `cfg_scale` or `scale`
- Use `samConfig` (camelCase) or `sam_config` (snake_case)

### Valid Scheduler Names

- `euler`
- `euler_ancestral`
- `dpm++_sde_karras` (default)
- `dpm++_2m_karras`
- `ddim`
- `pndm`
- `lms`
- `heun`
- `dpm_solver++`
- `unipc`

### LoRA Usage

#### Method 1: Via samConfig

```json
{
  "model": "sd/stable-diffusion-v1-5",
  "messages": [{"role": "user", "content": "a cat"}],
  "samConfig": {
    "steps": 25,
    "guidance_scale": 7.5,
    "lora_paths": ["my-lora-file"],
    "lora_scales": [0.8]
  }
}
```

#### Method 2: Embed in Prompt (A1111 style)

```json
{
  "model": "sd/stable-diffusion-v1-5",
  "messages": [{"role": "user", "content": "a cat <lora:my-lora-file:0.8>"}],
  "samConfig": {
    "steps": 25,
    "guidance_scale": 7.5
  }
}
```

### Examples

#### cURL (Text-to-Image)

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sd/stable-diffusion-v1-5",
    "messages": [{"role": "user", "content": "a cat"}],
    "samConfig": {
      "steps": 30,
      "guidance_scale": 10.0,
      "width": 512,
      "height": 512
    }
  }'
```

#### cURL (Image-to-Image with Qwen)

```bash
# Encode input image to base64
INPUT_IMAGE=$(base64 -i input.jpg)

curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"sd/qwen-image-edit-2511-Q2_K\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Transform it into Pixar-inspired 3D\"}],
    \"samConfig\": {
      \"steps\": 30,
      \"guidance_scale\": 2.5,
      \"width\": 1024,
      \"height\": 1024,
      \"scheduler\": \"euler\",
      \"strength\": 0.75,
      \"input_images\": [\"${INPUT_IMAGE}\"]
    }
  }"
```

#### Python

```python
import requests

response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    json={
        "model": "sd/stable-diffusion-v1-5",
        "messages": [{"role": "user", "content": "a beautiful landscape"}],
        "samConfig": {
            "steps": 30,
            "guidance_scale": 10.0,
            "width": 768,
            "height": 512,
            "negative_prompt": "ugly, blurry, low quality",
            "seed": 42,
            "scheduler": "euler_ancestral"
        }
    }
)

result = response.json()
image_url = result["choices"][0]["message"]["image_urls"][0]
print(f"Generated image: {image_url}")
```

#### JavaScript / TypeScript

```javascript
const response = await fetch('http://localhost:8080/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    model: 'sd/stable-diffusion-v1-5',
    messages: [{ role: 'user', content: 'a cat' }],
    samConfig: {
      steps: 30,
      guidance_scale: 10.0,
      width: 512,
      height: 512
    }
  })
});

const result = await response.json();
console.log(result.choices[0].message.image_urls[0]);
```

### Response Format

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
      "content": "1 image generated successfully.",
      "image_urls": ["http://localhost:8080/images/abc123.png"],
      "metadata": {
        "prompt": "a cat",
        "negative_prompt": "ugly, blurry",
        "steps": 30,
        "guidance_scale": 10.0,
        "seed": 12345,
        "model": "stable-diffusion-v1-5",
        "scheduler": "euler_ancestral",
        "width": 512,
        "height": 768
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

### Notes

1. **Field names are case-sensitive** - use exact names as documented
2. `samConfig` and `sam_config` are both accepted (Pydantic alias support)
3. All `samConfig` fields are optional - server defaults from `config.yaml` if not provided
4. Width and height are automatically rounded to nearest multiple of 8 (required for SD VAE)
5. `guidance_scale` can be set to `0.0` for SD Turbo and similar models
6. The actual generation parameters used are returned in the response `metadata`
7. Check server logs for debugging: `sudo journalctl -u alice.service -f`

### Troubleshooting

**Problem**: My `steps` or `guidance_scale` values are ignored

**Solution**: 
- Verify you're using `steps` (not `num_inference_steps`)
- Verify you're using `guidance_scale` (not `cfg_scale`)
- Check that `samConfig` is properly nested in the request
- Check server logs to see what values were received

**Problem**: Invalid field errors

**Solution**:
- Field names are case-sensitive
- Use `samConfig` (camelCase) or `sam_config` (snake_case)
- Ensure JSON is properly formatted

**Problem**: Images don't match expected parameters

**Solution**:
- Check the `metadata` field in the response to see actual values used
- Verify your client is sending the correct JSON structure
- Enable debug logging in `config.yaml` and check logs
