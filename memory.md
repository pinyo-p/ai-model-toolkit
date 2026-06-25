# AI Toolkit - Project Memory

> Hindsight memory bankId: `ai-toolkit`

## Architecture

- **Backend**: FastAPI (`main.py`) + `core/` modules (sdxl.py, flux2.py, zimage.py, lora.py, caption.py, image.py, gpu.py)
- **Frontend**: Single-page app (`static/index.html`)
- **Target**: NVIDIA GB10 (unified memory 128GB, `/home/yokiz/ai-model-toolkit/`)
- **Python**: 3.12 (GB10), venv at `/home/yokiz/ai-model-toolkit/venv`

## Supported Models

| Family | Variants | Pipeline | Text Encoder | Steps | CFG |
|--------|----------|----------|--------------|-------|-----|
| SDXL | Base 1.0, Pony, Illustrious | StableDiffusionXLPipeline | CLIP | 20 | 7.0 |
| FLUX.2[k] | Klein 9B | Flux2KleinPipeline | Qwen3 | 4 | 1.0 (forced) |
| FLUX.2[D] | Dev 32B | Flux2Pipeline | Mistral3 | 28 | 4.0 |
| z-Image | Base, Turbo | ZImagePipeline | Qwen3 | 9 | 0.0 |

## Key Technical Decisions

### FLUX.2 Pipeline Assembly (`core/flux2.py`)
- HF repo: `black-forest-labs/FLUX.2-klein-9B` (Klein), `black-forest-labs/FLUX.2-dev` (Dev)
- Always use `from_pretrained` for transformer (not `from_config`) — needed for correct `norm_out.linear` weights
- **CRITICAL**: `final_layer.adaLN_modulation.1.weight` → `norm_out.linear.weight` mapping is WRONG (cosine=-0.000320). Skip this mapping, keep pretrained weights.
- Scheduler: `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False)` — linear sigmas
- VAE decode: cast latents to float32 before decode (bfloat16 causes dtype mismatch)
- Klein detection: filename contains "klein" or "schnell"
- Dev detection: filename contains "dev"
- Weight remap: `_remap_flux2_state_dict()` — 201 checkpoint keys → 233 model keys (QKV split [q,k,v])

### Model Detection (`core/sdxl.py`)
- `_detect_model_type()`: name heuristic first, then tensor key inspection
- Family: zimage → flux2 → flux → sdxl → sd15

### CFG Handling (`core/sdxl.py`)
- Klein: force `guidance_scale=1.0` regardless of user setting
- Dev: use user's cfg value (default 4.0)

### Frontend (`static/index.html`)
- Auto-detect Klein vs Dev from model path for default steps/cfg
- Model list scans nested subdirectories (2 levels deep)

## Known Issues / Gotchas

- `FrozenDict` prevents modifying scheduler config in-place
- `pipe.scheduler = new_scheduler` after pipeline construction doesn't take effect
- FLUX.2 has no `negative_prompt` parameter
- Dev 32B needs ~64GB VRAM (bfloat16)
- `from_single_file` doesn't work for FLUX.2 checkpoints (always falls back to manual assembly)

## Session History

### 2026-06-25: FLUX.2 Generation Fix + Dev Support
- Fixed FLUX.2[k] noise increasing at every step (scheduler was using dynamic shifting)
- Fixed `norm_out.linear.weight` wrong mapping (cosine=-0.000320)
- Added FLUX.2[D] (Dev 32B) support
- Added nested model directory scanning
- All models tested: SDXL, FLUX.2[k], FLUX.2[D], z-Image-Base, z-Image-Turbo
