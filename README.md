# AI Toolkit

FastAPI web UI for SDXL image generation, LoRA training, captioning, and more.

> **Note:** This project was built with AI assistance. Code may not be perfect and could use improvement.

## Features

| Tab | Feature | Description |
|-----|---------|-------------|
| Generate | **SDXL Generate** | Image generation with LoRA, prompt, negative prompt, steps, seed, resolution |
| Batch | **Batch Generate** | Multiple prompts → ZIP download |
| Train | **Train LoRA** | 5-50 images, auto-caption (BLIP + metadata), select base model from local models |
| Train | **Image to LoRA** | Quick LoRA from 1-3 images |
| Merge | **Merge LoRA** | Merge multiple LoRA with weights |
| Merge | **Extract LoRA** | Extract LoRA from checkpoint |
| Load Model | **Download** | HuggingFace, CivitAI, or direct URL |
| Load Model | **Manage** | List/delete local models |
| Tools | **Caption** | BLIP image captioning |
| Tools | **Upscale** | 1-4x upscale (OpenCV) |
| Tools | **LoRA Info** | Inspect LoRA metadata |
| Settings | **Config** | HuggingFace/CivitAI tokens, models path |
| Settings | **Auth** | User management, change password |

## Requirements

- Python 3.10+
- CUDA recommended (CPU fallback available)
- ~2GB disk for base SDXL model

## Quick Start

### Linux / macOS

```bash
git clone https://github.com/pinyo-p/ai-model-toolkit.git ai-toolkit
cd ai-toolkit
chmod +x install.sh start.sh update.sh
./install.sh
./start.sh
```

### Windows

```cmd
git clone https://github.com/pinyo-p/ai-model-toolkit.git ai-toolkit
cd ai-toolkit
install.bat
start.bat
```

### Manual Install

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
python main.py
```

Open **http://localhost:7800**

Default login: `admin` / `admin` (change in Settings)

## Scripts

| Script | Linux/macOS | Windows | Description |
|--------|-------------|---------|-------------|
| Install | `install.sh` | `install.bat` | Create venv + install deps |
| Start | `start.sh` | `start.bat` | Activate venv + run server |
| Update | `update.sh` | `update.bat` | Git pull + install new deps |

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
├── install.sh           # Linux/macOS installer
├── install.bat          # Windows installer
├── start.sh             # Linux/macOS starter
├── start.bat            # Windows starter
├── update.sh            # Linux/macOS updater
└── update.bat           # Windows updater
```
