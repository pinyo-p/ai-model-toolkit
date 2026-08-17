# AI Toolkit

> ⚠️ **⚠️ WARNING — `scripts/finetune_llm_lora.py` IS UNDER DEVELOPMENT, NOT READY FOR USE ⚠️**  
> This script is experimental and may contain bugs. Do not use in production.  
> ⚠️ **⚠️ สคริป `scripts/finetune_llm_lora.py` กำลังพัฒนา ยังไม่พร้อมใช้งาน ⚠️**

FastAPI web UI for image generation (SDXL, FLUX.2[k]/[D], z-Image, Krea 2), deterministic LoRA evaluation, training prototypes, captioning, and more.

> **Note:** This project was built with AI assistance. Code may not be perfect and could use improvement.

## Supported Models

| Model Family | Variants | Default Steps | Default CFG | Notes |
|---|---|---|---|---|
| **SDXL** | Base 1.0, Pony, Illustrious | 20 | 7.0 | Full LoRA support |
| **FLUX.2[k]** | Klein 9B | 4 | 1.0 | Fast, distilled, CFG forced to 1.0 |
| **FLUX.2[D]** | Dev 32B | 28 | 4.0 | High quality, needs ~64GB VRAM |
| **z-Image** | Base, Turbo | 9 | 0.0 | Turbo recommended |
| **Krea 2** | Raw | 52 | 3.5 | LoRA training/base workflow; full Diffusers repository required |
| **Krea 2** | Turbo | 8 | 0.0 | Fast inference and LoRA testing; CFG forced to 0 |

### Krea 2 setup

Krea 2 support uses `Krea2Pipeline` from the pinned Diffusers source build in
`requirements.txt`. Run `./update.sh` (or `update.bat`) after pulling this version.

The simplest path is **Model → Custom** and enter `krea/Krea-2-Turbo` (or
`krea/Krea-2-Raw`); Diffusers will cache the complete repository. A local full
repository folder in the configured models directory is also selectable. A standalone
`raw.safetensors` or `turbo.safetensors` is not sufficient for this runtime because
the pipeline also needs its model index, Qwen3-VL text encoder, tokenizer, and
Qwen-Image VAE components. Hugging Face access requires accepting the model license
and configuring `HF_TOKEN` in Settings.

## Features

| Tab | Feature | Description |
|-----|---------|-------------|
| Generate | **Generate** | Image generation with LoRA, prompt, negative prompt, steps, seed, resolution |
| Test LoRA | **LoRA Evaluation** | Deterministic Prompt × Variant grid with matched seeds, per-variant settings, progress/cancel, and a downloadable test manifest |
| Train | **Train LoRA (prototype)** | Current UI scaffold for dataset training; the training engine is not production-ready yet |
| Train | **Image to LoRA (prototype)** | Planned quick workflow that expands 1-5 source images into a reviewed training dataset |
| Merge | **Merge LoRA** | Merge multiple LoRA with weights |
| Merge | **Extract LoRA** | Extract LoRA from checkpoint |
| Load Model | **File Manager** | Browse directories, upload (drag & drop), rename, create/delete dirs and files |
| Load Model | **File Details** | View file info (size, date, type) + secure download URL with auth |
| Load Model | **Download** | HuggingFace, CivitAI, or direct URL with optional save-as name + auto rename on conflict (-N suffix) |
| Tools | **Caption** | BLIP image captioning |
| Tools | **Upscale** | 1-4x upscale (OpenCV) |
| Tools | **LoRA Info** | Inspect LoRA metadata |
| Settings | **Config** | HuggingFace/CivitAI tokens, models path, base URL |
| Settings | **Auth** | User management, change password |

## Requirements

- Python 3.10+
- CUDA recommended (CPU fallback available)
- Disk and VRAM requirements vary significantly by family; Krea 2 is a large (~12B parameter) pipeline

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
| POST | `/api/comparison_generate_async` | Start a deterministic LoRA evaluation job |
| GET | `/api/comparison_progress` | Poll LoRA evaluation progress |
| GET | `/api/comparison_result` | Get a completed evaluation grid + reproducible manifest |
| POST | `/api/comparison_cancel` | Cancel an active evaluation |
| POST | `/api/image2lora` | LoRA from 1-3 images |
| POST | `/api/train_lora` | LoRA from 5-50 images |
| POST | `/api/merge_lora` | Merge LoRA files |
| POST | `/api/lora_info` | LoRA metadata |
| POST | `/api/extract_lora` | Extract LoRA from ckpt |
| POST | `/api/caption` | BLIP captioning |
| POST | `/api/auto_caption` | Auto caption with metadata fallback |
| POST | `/api/upscale` | Image upscale |
| GET | `/api/models` | List local models (scans nested subdirs) |
| DELETE | `/api/models` | Delete model |
| GET | `/api/models/browse` | Browse directory (name, size, type, modified) |
| POST | `/api/models/upload` | Upload files (drag & drop, multi-file) |
| POST | `/api/models/rename` | Rename file/directory |
| POST | `/api/models/directories` | List / create directories |
| GET | `/api/models/download` | Download file with auth + path traversal protection |
| POST | `/api/download_model` | Download model from URL (HuggingFace/CivitAI/Other) |
| GET/POST | `/api/settings` | App settings (including base URL) |
| POST | `/api/change_password` | Change credentials |

## Tech Stack

- Python 3.10+, FastAPI, Uvicorn
- PyTorch 2.9.0 + CUDA 12.8
- Diffusers source build (Krea 2, FLUX.2, SDXL), Transformers (Qwen3-VL, Qwen3, Mistral3)
- PEFT, Safetensors
- OpenCV (upscale), Pillow

## Project Structure

```
ai-toolkit/
├── main.py              # FastAPI app, all endpoints (browse/upload/rename/download)
├── core/
│   ├── sdxl.py          # Generation dispatch + deterministic LoRA evaluation
│   ├── runtimes.py      # Model-family detection, loaders, and generation defaults
│   ├── flux2.py         # FLUX.2[k]/[D] generation (Klein + Dev pipelines)
│   ├── zimage.py        # z-Image-Base/Turbo generation
│   ├── lora.py          # LoRA train/merge/extract
│   ├── caption.py       # BLIP captioning
│   ├── image.py         # Upscale
│   ├── gpu.py           # GPU detection
│   └── utils.py         # Helpers
├── static/
│   └── index.html       # Web UI (single-page, 7 tabs + modals)
├── requirements.txt
├── Dockerfile
├── install.sh / .bat    # Linux/macOS / Windows installer
├── start.sh / .bat      # Linux/macOS / Windows starter
└── update.sh / .bat     # Linux/macOS / Windows updater
```
