# AI Toolkit

> ⚠️ **⚠️ WARNING — `scripts/finetune_llm_lora.py` IS UNDER DEVELOPMENT, NOT READY FOR USE ⚠️**  
> This script is experimental and may contain bugs. Do not use in production.  
> ⚠️ **⚠️ สคริป `scripts/finetune_llm_lora.py` กำลังพัฒนา ยังไม่พร้อมใช้งาน ⚠️**

FastAPI web UI for image generation (SDXL, FLUX.2[k]/[D], z-Image, Krea 2), deterministic LoRA evaluation, Krea 2 LoRA training, captioning, and more.

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

Dataset training uses `krea/Krea-2-Raw` and writes the finished adapter under
`<models_path>/lora/krea2/`, ready to select in **Test LoRA** with Turbo. The first
run downloads the official Diffusers Krea 2 training script pinned to the same exact
source revision as the runtime and verifies its SHA-256 checksum before execution.
Training requires an NVIDIA CUDA GPU. A blank dataset trigger word is generated
automatically; all images must be captioned and 1–5 image datasets must first be
expanded and reviewed.

## Features

| Tab | Feature | Description |
|-----|---------|-------------|
| Generate | **Generate** | Image generation with LoRA, prompt, negative prompt, steps, seed, resolution |
| Test LoRA | **LoRA Evaluation** | Deterministic Prompt × Variant grid with matched seeds, per-variant settings, progress/cancel, and a downloadable test manifest |
| Test LoRA | **Dataset handoff** | One click prepares six concept-aware prompts and Base/LoRA 0.7/LoRA 1.0 Krea 2 Turbo variants from the latest completed training run |
| Test LoRA | **Human verdict** | Pick one winner per prompt; validated summaries are saved to the originating dataset without synthetic image scoring |
| Dataset | **Dataset Workspace** | Persistent preparation flow for 1–5 source images or full datasets up to 2,000 images, with later image imports |
| Dataset | **Validation & readiness** | Rejects invalid images, skips exact duplicates, reports resolution/aspect/caption coverage, and recommends the next action |
| Dataset | **Dataset curation** | Add images later or remove selected images and caption sidecars with rollback-safe manifest updates |
| Dataset | **Editable settings** | Rename a workspace or change its LoRA type and trigger word, with training-relevant edits tracked as a new revision |
| Dataset | **Portable export** | Download original images, caption sidecars, ImageFolder `metadata.jsonl`, and a history-free portable manifest as a ZIP |
| Dataset | **Portable restore** | Validate and restore an exported ZIP as a new workspace with its images, captions, LoRA type, and trigger word |
| Dataset | **ImageFolder import** | Import external ZIP datasets with nested images and matching `.txt` sidecars or root `metadata.jsonl` captions |
| Dataset | **Caption workspace** | Background metadata/BLIP captioning, editable captions, progress/cancel, and trainer-ready `.txt` sidecars |
| Dataset | **Large-dataset review** | Search, filter, and page through up to 2,000 images while retaining unsaved caption drafts across views |
| Dataset | **Bulk caption drafts** | Prepend, append, find/replace, or clear captions across selected images or the current filtered result, then review before saving |
| Dataset | **User-controlled sensitive label** | Optional metadata with local preview blurring; no content scanning, prompt rewriting, censorship, or workflow blocking |
| Dataset | **Technical quality audit** | Background, cancellable review hints for possible blur, extreme lighting, low contrast, and extreme aspect ratios; never removes or blocks images |
| Dataset | **Reference expansion** | Turn 1–5 source images into reviewable Qwen Image Edit candidates using concept-specific recipes; accepted images alone enter the dataset |
| Dataset | **Krea 2 LoRA training** | Fast/Balanced/Quality recipes, official Diffusers trainer, live step/loss progress, cancellation, and automatic LoRA registration |
| Dataset | **Training diagnostics** | Bounded live loss curves saved with each run for troubleshooting, while Test LoRA remains the quality verdict |
| Dataset | **Restart recovery** | Marks orphaned active training runs as interrupted on startup, preserves output/checkpoints, and reloads their profile and seed for an explicit retry |
| Dataset | **Dataset revisions** | Tracks the exact dataset revision and image count used by each LoRA, and warns when later image or caption changes make a run historical |
| Dataset | **Training history** | Compare every training attempt, its recipe/progress/file and linked human verdict; re-test any completed run instead of only the latest |
| Dataset | **Evaluation history** | Reopen the saved image/seed grid as read-only evidence, review Base/LoRA results, or load the winning generation settings in one click |
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
- Reference expansion downloads `Qwen/Qwen-Image-Edit-2509` on first use and requires CUDA; the editor is released after each batch before training

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
| POST | `/api/image2lora` | Legacy experimental LoRA endpoint (not production-ready) |
| POST | `/api/train_lora` | Legacy experimental LoRA endpoint (not production-ready) |
| GET/POST | `/api/datasets` | List or create persistent dataset workspaces |
| GET | `/api/datasets/{id}` | Dataset manifest, image metadata, readiness, and captions |
| PUT | `/api/datasets/{id}` | Update dataset name, LoRA type, and trigger word with revision-aware semantics |
| POST | `/api/datasets/{id}/export` | Prepare a portable dataset ZIP and return a short-lived one-time download URL |
| POST | `/api/datasets/import` | Import a portable export or external ImageFolder ZIP as a new workspace |
| POST | `/api/datasets/{id}/images` | Add validated images to an existing dataset and skip exact duplicates |
| DELETE | `/api/datasets/{id}/images` | Remove selected images and caption sidecars while keeping at least one source image |
| PUT | `/api/datasets/{id}/captions` | Save caption edits and matching `.txt` sidecars |
| POST | `/api/datasets/{id}/auto-caption` | Start background captioning for missing captions |
| GET | `/api/datasets/caption-progress/{job_id}` | Poll dataset caption progress |
| POST | `/api/datasets/caption-cancel/{job_id}` | Cancel dataset captioning |
| POST | `/api/datasets/{id}/quality-audit` | Start a non-blocking technical image quality audit |
| GET | `/api/datasets/quality-progress/{job_id}` | Poll quality audit progress and final review flags |
| POST | `/api/datasets/quality-cancel/{job_id}` | Cancel an audit after its current image |
| POST | `/api/datasets/{id}/expansion` | Start reference-guided dataset candidate generation |
| GET | `/api/expansion/{job_id}` | Poll expansion progress and generated candidate metadata |
| POST | `/api/expansion/{job_id}/cancel` | Cancel expansion at the next diffusion step |
| GET | `/api/datasets/{id}/expansion/{run_id}/candidates/{candidate_id}` | View an authenticated candidate image |
| POST | `/api/datasets/{id}/expansion/{run_id}/review` | Accept or reject selected candidates |
| GET | `/api/datasets/{id}/training-recipe` | Preview a Fast/Balanced/Quality Krea 2 recipe |
| POST | `/api/datasets/{id}/training` | Start official Krea 2 Raw LoRA training |
| GET | `/api/training/{job_id}` | Poll training steps, loss, log, and output path |
| POST | `/api/training/{job_id}/cancel` | Stop the trainer process group safely |
| GET | `/api/datasets/{id}/evaluation-preset` | Build a deterministic Test LoRA preset from a completed dataset training run |
| POST | `/api/datasets/{id}/evaluation-verdict` | Validate and save human winner picks for a completed LoRA evaluation |
| GET | `/api/datasets/{id}/evaluations/{evaluation_id}/images/{filename}` | Read a dataset-owned evaluation evidence image with authentication |
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
│   ├── datasets.py      # Persistent dataset manifests, validation, captions, readiness
│   ├── expansion.py     # Qwen Image Edit candidate recipes and inference adapter
│   ├── evaluation_presets.py # Dataset-aware Krea 2 Turbo test prompts and variants
│   ├── training.py      # Simple recipes + pinned official Krea 2 trainer adapter
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
