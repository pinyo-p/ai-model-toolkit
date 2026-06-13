# AI Toolkit for Nvidia GB10 ARM64 + CUDA 12.8

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Server will start at `http://0.0.0.0:7800`

## API Endpoints

| Method | Endpoint | Function |
|--------|----------|----------|
| GET | /api/gpu | check_gpu |
| POST | /api/caption | image_captioning |
| POST | /api/train_lora | image2lora |
| POST | /api/merge_lora | lora_merge |
| POST | /api/generate | sdxl_generate |
| POST | /api/batch_generate | batch_generate |
| POST | /api/lora_info | lora_info |
| POST | /api/extract_lora | extract_lora |
| POST | /api/upscale | upscale |

## Features

- SDXL image generation with LoRA support
- LoRA training from 1-5 images
- LoRA merging and extraction
- Image captioning with BLIP
- Image upscaling with Real-ESRGAN
- Batch generation with ZIP download

## Tech Stack

- Python 3.12 + FastAPI + Uvicorn
- PyTorch 2.8.0 + CUDA 12.8
- Diffusers 0.30.3
- Transformers 4.44.2