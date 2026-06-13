import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import os
from .gpu import check_gpu


_pipeline = None


def _get_pipeline(lora_path: str = None):
    global _pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    if _pipeline is None:
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=dtype
        )
        _pipeline = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            vae=vae,
            torch_dtype=dtype,
        )

        gpu_info = check_gpu()
        if gpu_info["vram_total_gb"] < 20:
            _pipeline.enable_vae_slicing()
            _pipeline.enable_vae_tiling()

        if device == "cuda":
            _pipeline.enable_model_cpu_offload()
        else:
            _pipeline = _pipeline.to("cpu")

    return _pipeline


def sdxl_generate(
    prompt: str,
    negative: str = "",
    lora_path: str = None,
    steps: int = 20,
    seed: int = 42,
    width: int = 1024,
    height: int = 1024
) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not torch.cuda.is_available() and device == "cpu":
        raise RuntimeError("CUDA not available, falling back to CPU")

    pipeline = _get_pipeline(lora_path)

    generator = torch.Generator(device=device).manual_seed(seed)

    extra_kwargs = {}
    if lora_path and os.path.exists(lora_path):
        from safetensors.torch import load_file
        lora_state_dict = load_file(lora_path)
        pipeline.load_lora_weights(os.path.dirname(lora_path) or ".")
        extra_kwargs["adapter_weights"] = [1.0]

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative,
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
    steps: int = 20,
    seed: int = 42
) -> list[Image.Image]:
    images = []
    for i, prompt in enumerate(prompts):
        img = sdxl_generate(
            prompt=prompt,
            negative=negative,
            lora_path=lora_path,
            steps=steps,
            seed=seed + i,
            width=1024,
            height=1024
        )
        images.append(img)
    return images