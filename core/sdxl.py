import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, AutoencoderKL
from PIL import Image
import os
from .gpu import check_gpu


_pipelines = {}


def _detect_model_type(model_path: str) -> str:
    model_lower = model_path.lower()
    if any(x in model_lower for x in ["flux", "z-image", "z_image"]):
        return "flux"
    if any(x in model_lower for x in ["xl", "sdxl", "pony", "sd_xl", "illustrious"]):
        return "sdxl"
    if any(x in model_lower for x in ["v1-5", "v1.5", "sd15", "sd-1", "runwayml"]):
        return "sd15"
    return "sdxl"


def _get_pipeline(
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_path: str = None,
    text_encoder_path: str = None,
):
    cache_key = f"{model_path}|{vae_path}|{text_encoder_path}"

    if cache_key in _pipelines:
        return _pipelines[cache_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    model_type = _detect_model_type(model_path)

    vae = None
    if vae_path and os.path.exists(vae_path):
        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)

    if model_type == "flux":
        try:
            from diffusers import FluxPipeline
            pipeline = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
        except Exception:
            from diffusers import StableDiffusionXLPipeline
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                vae=vae,
                torch_dtype=dtype,
            )
    elif model_type == "sd15":
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_path,
            vae=vae,
            torch_dtype=dtype,
        )
    else:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_path,
            vae=vae,
            torch_dtype=dtype,
        )

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
    lora_path: str = None,
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_path: str = None,
    text_encoder_path: str = None,
    steps: int = 20,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024
) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _get_pipeline(model_path, vae_path, text_encoder_path)

    generator = torch.Generator(device=device).manual_seed(seed)

    if lora_path and os.path.exists(lora_path):
        pipeline.load_lora_weights(os.path.dirname(lora_path) or ".")

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative if negative else None,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=7.0,
    ).images[0]

    return image


def batch_generate(
    prompts: list[str],
    negative: str = "",
    lora_path: str = None,
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_path: str = None,
    text_encoder_path: str = None,
    steps: int = 20,
    seed: int = 42
) -> list[Image.Image]:
    images = []
    for i, prompt in enumerate(prompts):
        img = sdxl_generate(
            prompt=prompt,
            negative=negative,
            lora_path=lora_path,
            model_path=model_path,
            vae_path=vae_path,
            text_encoder_path=text_encoder_path,
            steps=steps,
            seed=seed + i,
            width=1024,
            height=1024
        )
        images.append(img)
    return images
