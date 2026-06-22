import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, AutoencoderKL
from fastapi import HTTPException
from PIL import Image
import os
import struct
import json
from .gpu import check_gpu

from safetensors.torch import load_file as safetensors_load_file


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
    # Fallback: name heuristic (check FIRST to avoid misclassification)
    model_lower = model_path.lower()
    if any(x in model_lower for x in ["z-image", "z_image", "zimage"]):
        return "zimage"
    if any(x in model_lower for x in ["flux2", "flux.2", "flux-2"]):
        return "flux2"
    if any(x in model_lower for x in ["flux"]):
        return "flux"
    if any(x in model_lower for x in ["xl", "sdxl", "pony", "sd_xl", "illustrious"]):
        return "sdxl"
    if any(x in model_lower for x in ["v1-5", "v1.5", "sd15", "sd-1", "runwayml"]):
        return "sd15"

    # Single safetensors file → read keys for detection
    if model_path.endswith('.safetensors') and os.path.isfile(model_path):
        fname = os.path.basename(model_path).lower()
        keys = _read_safetensors_meta(model_path)
        if keys:
            joined = ' '.join(k.lower() for k in keys)
            if 'single_stream_blocks' in joined and 'double_stream' not in joined:
                return "zimage"
            if 'noise_refiner' in joined or 'cap_embedder' in joined or 'context_refiner' in joined:
                return "zimage"
            if 'mmdit.' in joined:
                return "sd3"
            # FLUX.2 / FLUX.1: model.diffusion_model.double_blocks + single_blocks
            if 'model.diffusion_model' in joined:
                has_double = 'double_blocks' in joined
                has_single = 'single_blocks' in joined
                has_sdxl = any(x in joined for x in ['input_blocks.', 'mid_block.', 'output_blocks.'])
                has_pixart = 'x_embedder' in joined and 'model.diffusion_model.layers.' in joined
                if has_double and has_single:
                    # FLUX.2 (both double & single blocks)
                    if any(x in fname for x in ['flux2', 'flux.2', 'flux-2']):
                        return "flux2"
                    return "flux2"
                if has_double:
                    return "flux"
                if has_sdxl:
                    return "sdxl"
                if has_pixart:
                    return "pixart"
                return "sdxl"
            if 'double_stream' in joined:
                if any(x in fname for x in ['flux2', 'flux.2', 'flux-2']):
                    return "flux2"
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
                    "Flux2Pipeline": "flux2",
                    "Flux2KleinPipeline": "flux2",
                    "Flux2KleinKVPipeline": "flux2",
                    "ZImagePipeline": "zimage",
                    "HunyuanDiTPipeline": "hunyuan",
                    "PixArtAlphaPipeline": "pixart",
                    "KolorsPipeline": "kolors",
                }
                return mapping.get(cls_name, "sdxl")
            except Exception:
                pass

    return "sdxl"


def _load_pipeline(pipeline_cls, model_path, vae=None, dtype=torch.float16, **extra):
    """Load a pipeline, using from_single_file for single files and from_pretrained for directories/HF IDs."""
    is_file = os.path.isfile(model_path) and not os.path.isdir(model_path)
    # Always pass HF token for gated repos
    token = os.environ.get("HF_TOKEN")
    if token:
        extra.setdefault('token', token)
    kwargs = dict(torch_dtype=dtype, **extra)
    if is_file:
        try:
            pipe = pipeline_cls.from_single_file(model_path, **kwargs)
        except AttributeError as e:
            if 'text_model' in str(e):
                pipe = _fallback_load_sdxl_from_file(model_path, dtype)
            else:
                raise
        except Exception as e:
            if 'text_model' in str(e):
                pipe = _fallback_load_sdxl_from_file(model_path, dtype)
            else:
                raise
    else:
        pipe = pipeline_cls.from_pretrained(model_path, **kwargs)

    if vae is not None:
        pipe.vae = vae
    return pipe


def _fallback_load_sdxl_from_file(model_path, dtype):
    """Fallback: load base SDXL pipeline from HF hub, then load UNet weights from checkpoint."""
    hf_token = os.environ.get("HF_TOKEN")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=dtype, token=hf_token
    )
    # Load checkpoint and filter UNet keys only
    ckpt = safetensors_load_file(model_path, device="cpu")
    unet_prefix = "model.diffusion_model."
    unet_state = {k.replace(unet_prefix, ""): v for k, v in ckpt.items() if k.startswith(unet_prefix)}
    if unet_state:
        pipe.unet.load_state_dict(unet_state, strict=False)
    del ckpt
    return pipe


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
        if os.path.isfile(vae_path) and vae_path.endswith('.safetensors'):
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)

    if model_type == "zimage":
        from diffusers import ZImagePipeline
        hf_token = os.environ.get("HF_TOKEN")
        zimage_repo = "Tongyi-MAI/Z-Image-Turbo"

        # HF repo ID → use from_pretrained directly (repo has everything)
        if not os.path.isfile(model_path) and not os.path.isdir(model_path):
            pipeline = ZImagePipeline.from_pretrained(
                model_path, torch_dtype=dtype, low_cpu_mem_usage=False, token=hf_token
            )
        else:
            # Local file → load VAE + text encoder from HF repo, then from_single_file
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load VAE from repo
            if vae is None:
                try:
                    vae = AutoencoderKL.from_pretrained(
                        zimage_repo, subfolder="vae", torch_dtype=dtype, token=hf_token
                    )
                except Exception:
                    pass

            # Load text encoder + tokenizer from repo
            # NOTE: tokenizer is in a SEPARATE tokenizer/ folder, not text_encoder/
            text_encoder = None
            tokenizer = None
            model_dir = os.path.dirname(model_path) if os.path.isfile(model_path) else ""

            # Try local paths first
            local_te_paths = [
                text_encoder_path,
                os.path.join(model_dir, "text_encoder"),
                os.path.join(model_dir, "phi"),
            ]
            for tp in local_te_paths:
                if tp and os.path.exists(tp):
                    try:
                        if os.path.isfile(tp) and tp.endswith('.safetensors'):
                            text_encoder = AutoModelForCausalLM.from_single_file(tp, torch_dtype=dtype)
                        else:
                            text_encoder = AutoModelForCausalLM.from_pretrained(tp, torch_dtype=dtype)
                        break
                    except Exception:
                        pass

            # Tokenizer: try local tokenizer/ dir first
            local_tok_paths = [
                os.path.join(model_dir, "tokenizer"),
                text_encoder_path,
                os.path.join(model_dir, "text_encoder"),
            ]
            for tok_p in local_tok_paths:
                if tok_p and os.path.exists(tok_p):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(tok_p, trust_remote_code=True)
                        if getattr(tokenizer, 'chat_template', None):
                            break
                    except Exception:
                        pass

            # Fallback: download from HF repo (text_encoder + tokenizer are separate subfolders)
            if text_encoder is None:
                try:
                    text_encoder = AutoModelForCausalLM.from_pretrained(
                        zimage_repo, subfolder="text_encoder", torch_dtype=dtype, token=hf_token
                    )
                except Exception:
                    pass
            if tokenizer is None or not getattr(tokenizer, 'chat_template', None):
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        zimage_repo, subfolder="tokenizer", trust_remote_code=True, token=hf_token
                    )
                except Exception:
                    pass

            kwargs = dict(dtype=dtype, low_cpu_mem_usage=False)
            if vae is not None:
                kwargs['vae'] = vae
            if text_encoder is not None:
                kwargs['text_encoder'] = text_encoder
            if tokenizer is not None:
                kwargs['tokenizer'] = tokenizer
            pipeline = _load_pipeline(ZImagePipeline, model_path, **kwargs)
            # Optional optimizations for Z-Image-Turbo
            if hasattr(pipeline, 'transformer'):
                try:
                    pipeline.transformer.compile()
                except Exception:
                    pass
    elif model_type == "pixart":
        try:
            from diffusers import PixArtAlphaPipeline
            pipeline = _load_pipeline(PixArtAlphaPipeline, model_path, dtype=dtype)
        except Exception:
            kwargs = dict(vae=vae, dtype=dtype)
            # Only use text_encoder if it's actually CLIP
            if text_encoder_path and os.path.exists(text_encoder_path):
                try:
                    from transformers import CLIPTextModel, CLIPTokenizer
                    text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=dtype)
                    kwargs['text_encoder'] = text_encoder
                except Exception:
                    pass
            pipeline = _load_pipeline(StableDiffusionXLPipeline, model_path, **kwargs)
    elif model_type == "flux":
        try:
            from diffusers import FluxPipeline
            pipeline = _load_pipeline(FluxPipeline, model_path, dtype=dtype)
        except Exception as e:
            if 'Mistral' in str(e) or 'text_model' in str(e) or 'Qwen' in str(e):
                # Try auto-detection via DiffusionPipeline
                try:
                    from diffusers import DiffusionPipeline
                    pipeline = _load_pipeline(DiffusionPipeline, model_path, dtype=dtype)
                except Exception:
                    raise HTTPException(status_code=400,
                        detail="This appears to be a FLUX.2 checkpoint. "
                               "Make sure you have the latest diffusers: pip install -U diffusers")
            else:
                kwargs = dict(vae=vae, dtype=dtype)
                if text_encoder_path and os.path.exists(text_encoder_path):
                    try:
                        from transformers import CLIPTextModel, CLIPTokenizer
                        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=dtype)
                        kwargs['text_encoder'] = text_encoder
                    except Exception:
                        pass
                pipeline = _load_pipeline(StableDiffusionXLPipeline, model_path, **kwargs)
    elif model_type == "flux2":
        if os.path.isfile(model_path):
            from diffusers import Flux2Pipeline, Flux2KleinPipeline, Flux2KleinKVPipeline, DiffusionPipeline
            last_err = None
            for cls in [Flux2KleinPipeline, Flux2KleinKVPipeline, Flux2Pipeline, DiffusionPipeline]:
                try:
                    pipeline = _load_pipeline(cls, model_path, dtype=dtype)
                    break
                except Exception as e:
                    last_err = e
                    continue
            else:
                raise HTTPException(status_code=400,
                    detail=f"Failed to load FLUX.2 checkpoint: {last_err}")
        else:
            from diffusers import DiffusionPipeline
            pipeline = _load_pipeline(DiffusionPipeline, model_path, dtype=dtype)
    elif model_type == "sd15":
        pipeline = _load_pipeline(StableDiffusionPipeline, model_path, dtype=dtype)
    else:
        # SDXL: load default VAE if not provided (some checkpoints don't include one)
        if vae is None:
            try:
                vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=dtype)
            except Exception:
                pass
        kwargs = dict(vae=vae, dtype=dtype)
        pipeline = _load_pipeline(StableDiffusionXLPipeline, model_path, **kwargs)

    gpu_info = check_gpu()
    if gpu_info["vram_total_gb"] < 20:
        if hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

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
    height: int = 1024,
    progress_cb=None,
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

    def _step_cb(pipeline, step_index, timestep, callback_kwargs):
        if progress_cb:
            try:
                progress_cb(step_index, steps)
            except Exception:
                pass
        return callback_kwargs

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative if negative else None,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=cfg,
        callback_on_step_end=_step_cb,
    ).images[0]

    return image


def sdxl_generate_parallel(
    prompts: list[str],
    negative: str = "",
    lora_paths: list = None,
    lora_weights: list = None,
    model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_path: str = None,
    text_encoder_path: str = None,
    steps: int = 20,
    cfg: float = 7.0,
    seeds: list[int] = None,
    width: int = 1024,
    height: int = 1024,
    progress_cb=None,
) -> list[Image.Image]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _get_pipeline(model_path, vae_path, text_encoder_path)

    if seeds is None:
        seeds = list(range(len(prompts)))

    generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

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

    def _step_cb(pipeline, step_index, timestep, callback_kwargs):
        if progress_cb:
            try:
                progress_cb(step_index, steps)
            except Exception:
                pass
        return callback_kwargs

    negative_prompts = [negative if negative else None] * len(prompts)

    result = pipeline(
        prompt=prompts,
        negative_prompt=negative_prompts,
        num_inference_steps=steps,
        generator=generators,
        width=width,
        height=height,
        guidance_scale=cfg,
        callback_on_step_end=_step_cb,
    )

    return result.images


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
