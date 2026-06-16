import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, AutoencoderKL
from fastapi import HTTPException
from PIL import Image
import os
import struct
import json
from .gpu import check_gpu


_pipelines = {}


def _read_safetensors_meta(path: str):
    try:
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            if header_len <= 0 or header_len > 50 * 1024 * 1024:
                return None
            raw = f.read(header_len)
            if len(raw) != header_len:
                return None
            header = json.loads(raw)
        return [k for k in header if k != "__metadata__"]
    except Exception:
        return None


def _detect_model_type(model_path: str) -> str:
    # Single safetensors file → read keys for detection
    if model_path.endswith('.safetensors') and os.path.isfile(model_path):
        keys = _read_safetensors_meta(model_path)
        if keys:
            joined = ' '.join(k.lower() for k in keys)
            if 'single_stream_blocks' in joined and 'double_stream' not in joined:
                return "zimage"
            if 'mmdit.' in joined:
                return "sd3"
            if 'model.diffusion_model' in joined:
                if any(x in joined for x in ['input_blocks.', 'mid_block.', 'output_blocks.']):
                    return "sdxl"
                # DiT wrapped under model.diffusion_model (PixArt-style)
                if 'x_embedder' in joined and 'model.diffusion_model.layers.' in joined:
                    return "pixart"
                return "sdxl"
            if 'double_stream' in joined:
                return "flux"
            if 'transformer_blocks' in joined and 'time_text_embed' in joined:
                return "flux"
            if 'transformer_blocks' in joined and ('attn1' in joined or 'attn2' in joined):
                return "pixart"
            if 'x_embedder' in joined and 'layers.' in joined:
                return "pixart"

    # Folder → check model_index.json
    if os.path.isdir(model_path):
        idx_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(idx_path):
            try:
                with open(idx_path) as f:
                    idx = json.load(f)
                cls_name = idx.get("_class_name", "")
                mapping = {
                    "StableDiffusionPipeline": "sd15",
                    "StableDiffusionXLPipeline": "sdxl",
                    "StableDiffusion3Pipeline": "sd3",
                    "FluxPipeline": "flux",
                    "Flux2Pipeline": "flux",
                    "ZImagePipeline": "zimage",
                    "HunyuanDiTPipeline": "hunyuan",
                    "PixArtAlphaPipeline": "pixart",
                    "KolorsPipeline": "kolors",
                }
                return mapping.get(cls_name, "sdxl")
            except Exception:
                pass

    # Fallback: name heuristic
    model_lower = model_path.lower()
    if any(x in model_lower for x in ["z-image", "z_image"]):
        return "zimage"
    if any(x in model_lower for x in ["flux"]):
        return "flux"
    if any(x in model_lower for x in ["xl", "sdxl", "pony", "sd_xl", "illustrious"]):
        return "sdxl"
    if any(x in model_lower for x in ["v1-5", "v1.5", "sd15", "sd-1", "runwayml"]):
        return "sd15"
    return "sdxl"


def _load_pipeline(pipeline_cls, model_path, vae=None, dtype=torch.float16, **extra):
    """Load a pipeline, using from_single_file for single files and from_pretrained for directories/HF IDs."""
    is_file = os.path.isfile(model_path) and not os.path.isdir(model_path)
    kwargs = dict(torch_dtype=dtype, **extra)
    if vae is not None:
        kwargs['vae'] = vae
    if is_file:
        try:
            return pipeline_cls.from_single_file(model_path, **kwargs)
        except AttributeError:
            cls_name = getattr(pipeline_cls, '__name__', str(pipeline_cls))
            raise HTTPException(status_code=400,
                detail=f"{cls_name}.from_single_file() not available in this diffusers version. "
                       f"Upgrade: pip install -U diffusers")
    return pipeline_cls.from_pretrained(model_path, **kwargs)


def _get_pipeline(
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_path: str = None,
    text_encoder_path: str = None,
):
    cache_key = f"{model_path}|{vae_path}|{text_encoder_path}"

    if cache_key in _pipelines:
        return _pipelines[cache_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = _detect_model_type(model_path)
    dtype = torch.bfloat16 if model_type == "zimage" else torch.float16

    vae = None
    if vae_path and os.path.exists(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)

    if model_type == "zimage":
        from diffusers import ZImagePipeline
        pipeline = _load_pipeline(ZImagePipeline, model_path, dtype=dtype, low_cpu_mem_usage=False)
    elif model_type == "pixart":
        from diffusers import PixArtAlphaPipeline
        pipeline = _load_pipeline(PixArtAlphaPipeline, model_path, dtype=dtype)
    elif model_type == "flux":
        try:
            from diffusers import FluxPipeline
            pipeline = _load_pipeline(FluxPipeline, model_path, dtype=dtype)
        except Exception:
            from diffusers import StableDiffusionXLPipeline
            pipeline = _load_pipeline(StableDiffusionXLPipeline, model_path, vae=vae, dtype=dtype)
    elif model_type == "sd15":
        pipeline = _load_pipeline(StableDiffusionPipeline, model_path, vae=vae, dtype=dtype)
    else:
        pipeline = _load_pipeline(StableDiffusionXLPipeline, model_path, vae=vae, dtype=dtype)

    if text_encoder_path and os.path.exists(text_encoder_path):
        try:
            from transformers import CLIPTextModel, CLIPTokenizer
            if hasattr(pipeline, "text_encoder"):
                pipeline.text_encoder = CLIPTextModel.from_pretrained(
                    text_encoder_path, torch_dtype=dtype
                )
        except Exception:
            pass

    gpu_info = check_gpu()
    if gpu_info["vram_total_gb"] < 20:
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

    if device == "cuda":
        if hasattr(pipeline, "enable_model_cpu_offload"):
            pipeline.enable_model_cpu_offload()
        else:
            pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to("cpu")

    _pipelines[cache_key] = pipeline
    return pipeline


def sdxl_generate(
    prompt: str,
    negative: str = "",
    lora_paths: list = None,
    lora_weights: list = None,
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_path: str = None,
    text_encoder_path: str = None,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024
) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _get_pipeline(model_path, vae_path, text_encoder_path)

    generator = torch.Generator(device=device).manual_seed(seed)

    if lora_paths:
        for i, (lp, lw) in enumerate(zip(lora_paths, lora_weights or [])):
            if lp and os.path.exists(lp):
                pipeline.load_lora_weights(
                    os.path.dirname(lp) or ".",
                    weight_name=os.path.basename(lp),
                    adapter_name=f"lora_{i}",
                )
        adapter_names = [f"lora_{i}" for i in range(len(lora_paths))]
        adapter_weights = lora_weights or [1.0] * len(lora_paths)
        pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative if negative else None,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=cfg,
    ).images[0]

    return image


def batch_generate(
    prompts: list[str],
    negative: str = "",
    lora_paths: list = None,
    lora_weights: list = None,
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_path: str = None,
    text_encoder_path: str = None,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = 42
) -> list[Image.Image]:
    images = []
    for i, prompt in enumerate(prompts):
        img = sdxl_generate(
            prompt=prompt,
            negative=negative,
            lora_paths=lora_paths,
            lora_weights=lora_weights,
            model_path=model_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            steps=steps,
            cfg=cfg,
            seed=seed + i,
            width=1024,
            height=1024
        )
        images.append(img)
    return images
