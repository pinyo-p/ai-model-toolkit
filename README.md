# AI Toolkit

FastAPI web UI for SDXL image generation, LoRA training, captioning, and more.

> **Note:** This project was built with AI assistance. Code may not be perfect and could use improvement.

## Features

- **Generate** - SDXL image generation with LoRA support, custom prompt, negative prompt, steps, seed, resolution
- **Batch Generate** - Generate multiple images from prompts, download as ZIP
- **Train LoRA** - Train LoRA from 5-50 images with auto-captioning (BLIP + metadata fallback), select base model from local models
- **Image to LoRA** - Quick LoRA from 1-3 images
- **Merge LoRA** - Merge multiple LoRA with weights
- **Extract LoRA** - Extract LoRA from checkpoint
- **Load Model** - Download models from HuggingFace, CivitAI, or direct URL; list/delete local models
- **Caption** - Generate image captions via BLIP
- **Upscale** - 1-4x upscale via OpenCV
- **LoRA Info** - Inspect LoRA metadata
- **Settings** - HuggingFace/CivitAI tokens, models path, user management

## Requirements

- Python 3.10+
- CUDA recommended (CPU fallback available)

## Installation

### Quick Install (Linux / macOS)

```bash
git clone <repo-url> ai-toolkit
cd ai-toolkit
chmod +x install.bash
./install.bash
```

### Manual Install

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Windows

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
# Activate venv first
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

python main.py
```

Open `http://localhost:7800` in your browser.

**Default login:** `admin` / `admin` (change in Settings)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/gpu` | GPU info (VRAM, CUDA) |
| POST | `/api/generate` | Generate image |
| POST | `/api/batch_generate` | Batch generate (ZIP) |
| POST | `/api/image2lora` | LoRA from 1-3 images |
| POST | `/api/train_lora` | LoRA from 5-50 images |
| POST | `/api/merge_lora` | Merge LoRA files |
| POST | `/api/lora_info` | LoRA metadata |
| POST | `/api/extract_lora` | Extract LoRA from ckpt |
| POST | `/api/caption` | BLIP captioning |
| POST | `/api/auto_caption` | Auto caption with metadata fallback |
| POST | `/api/upscale` | Image upscale |
| GET | `/api/models` | List local models |
| DELETE | `/api/models` | Delete model |
| POST | `/api/download_model` | Download model from URL |
| GET/POST | `/api/settings` | App settings |
| POST | `/api/change_password` | Change credentials |

## Tech Stack

- Python 3.10+, FastAPI, Uvicorn
- PyTorch 2.9.0 + CUDA 12.8
- Diffusers, Transformers, Accelerate
- PEFT, Safetensors
- OpenCV (upscale), Pillow

## Project Structure

```
ai-toolkit/
├── main.py              # FastAPI app, all endpoints
├── core/
│   ├── sdxl.py          # Image generation
│   ├── lora.py          # LoRA train/merge/extract
│   ├── caption.py       # BLIP captioning
│   ├── image.py         # Upscale
│   ├── gpu.py           # GPU detection
│   └── utils.py         # Helpers
├── static/
│   └── index.html       # Web UI (single-page)
├── requirements.txt
├── Dockerfile
└── install.bash
```
