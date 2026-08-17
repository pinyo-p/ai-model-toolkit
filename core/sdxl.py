import torch
import inspect
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image
import os
import threading
from functools import wraps

from safetensors.torch import load_file as safetensors_load_file
from .runtimes import RuntimeLoadContext, detect_model_type, get_runtime


_pipelines = {}
_inference_lock = threading.RLock()


class CancelGeneration(Exception):
    pass


def _serialized_inference(function):
    """Protect cached pipelines and their mutable LoRA state across background jobs."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with _inference_lock:
            return function(*args, **kwargs)
    return wrapped


def _reset_pipeline_loras(pipeline):
    """Remove adapters loaded by a previous generation from a cached pipeline."""
    had_adapters = bool(getattr(pipeline, "_ai_toolkit_lora_adapters", []))

    if hasattr(pipeline, "unload_lora_weights"):
        try:
            pipeline.unload_lora_weights()
            pipeline._ai_toolkit_lora_adapters = []
            return
        except Exception:
            # Some pipeline implementations expose the method but do not support it.
            pass

    if had_adapters and hasattr(pipeline, "delete_adapters"):
        try:
            pipeline.delete_adapters(pipeline._ai_toolkit_lora_adapters)
            pipeline._ai_toolkit_lora_adapters = []
            return
        except Exception:
            pass

    if had_adapters:
        raise RuntimeError(
            "This pipeline cannot remove the LoRA from the previous run. "
            "Unload the model before evaluating another variant."
        )

    if hasattr(pipeline, "disable_lora"):
        try:
            pipeline.disable_lora()
        except Exception:
            pass


def _configure_pipeline_loras(pipeline, lora_paths=None, lora_weights=None):
    """Reset cached adapter state, then load exactly the requested LoRAs."""
    _reset_pipeline_loras(pipeline)

    requested = []
    supplied_weights = list(lora_weights or [])
    for index, path in enumerate(lora_paths or []):
        if path and os.path.exists(path):
            weight = supplied_weights[index] if index < len(supplied_weights) else 1.0
            requested.append((path, float(weight)))

    if not requested:
        return []
    if not hasattr(pipeline, "load_lora_weights") or not hasattr(pipeline, "set_adapters"):
        raise RuntimeError("The selected model pipeline does not support LoRA adapters.")

    adapter_names = []
    adapter_weights = []
    for index, (path, weight) in enumerate(requested):
        adapter_name = f"ai_toolkit_lora_{index}"
        pipeline.load_lora_weights(
            os.path.dirname(path) or ".",
            weight_name=os.path.basename(path),
            adapter_name=adapter_name,
            ignore_mismatched_sizes=True,
        )
        adapter_names.append(adapter_name)
        adapter_weights.append(weight)

    pipeline.set_adapters(adapter_names, adapter_weights=adapter_weights)
    pipeline._ai_toolkit_lora_adapters = adapter_names
    return adapter_names


def _detect_model_type(model_path: str) -> str:
    return detect_model_type(model_path)


def _load_pipeline(pipeline_cls, model_path, vae=None, dtype=torch.float16, **extra):
    """Load a pipeline, using from_single_file for single files and from_pretrained for directories/HF IDs."""
    is_file = os.path.isfile(model_path) and not os.path.isdir(model_path)
    # Always pass HF token for gated repos
    token = os.environ.get("HF_TOKEN")
    if token:
        extra.setdefault('token', token)
    kwargs = dict(torch_dtype=dtype, **extra)
    if vae is not None:
        kwargs['vae'] = vae
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
    on_message=None,
    on_progress=None,
):
    cache_key = f"{model_path}|{vae_path}|{text_encoder_path}"

    if cache_key in _pipelines:
        return _pipelines[cache_key]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = _detect_model_type(model_path)
    runtime = get_runtime(model_type)
    dtype = runtime.dtype

    vae = None
    if model_type != "krea2" and vae_path and os.path.exists(vae_path):
        if os.path.isfile(vae_path) and vae_path.endswith('.safetensors'):
            vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
        else:
            vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)

    pipeline = runtime.load(RuntimeLoadContext(
        model_path=model_path,
        vae_path=vae_path,
        text_encoder_path=text_encoder_path,
        vae=vae,
        dtype=dtype,
        load_pipeline=_load_pipeline,
        on_message=on_message,
        on_progress=on_progress,
    ))

    if device == "cuda":
        pipeline = pipeline.to(device)
    else:
        pipeline = pipeline.to("cpu")

    # Speed optimizations
    if device == "cuda":
        for comp_name in ("transformer", "unet"):
            comp = getattr(pipeline, comp_name, None)
            if comp is not None:
                try:
                    comp.to(memory_format=torch.channels_last)
                except Exception:
                    pass
                try:
                    if hasattr(comp, "fuse_qkv_projections"):
                        comp.fuse_qkv_projections()
                except Exception:
                    pass
        try:
            vae_comp = pipeline.vae
            vae_comp.to(memory_format=torch.channels_last)
        except Exception:
            pass

    _pipelines[cache_key] = pipeline
    return pipeline


@_serialized_inference
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
    cancel_event=None,
    on_message=None,
    on_progress=None,
    configure_loras: bool = True,
) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _get_pipeline(model_path, vae_path, text_encoder_path, on_message=on_message, on_progress=on_progress)

    generator = torch.Generator(device=device).manual_seed(seed)

    if configure_loras:
        _configure_pipeline_loras(pipeline, lora_paths, lora_weights)

    def _step_cb(pipeline, step_index, timestep, callback_kwargs):
        if cancel_event and cancel_event.is_set():
            raise CancelGeneration()
        if progress_cb:
            try:
                progress_cb(step_index, steps)
            except Exception:
                pass
        return callback_kwargs

    runtime = get_runtime(_detect_model_type(model_path))
    effective_cfg = runtime.effective_cfg(model_path, cfg)

    call_kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=effective_cfg,
        callback_on_step_end=_step_cb,
    )
    if negative and 'negative_prompt' in inspect.signature(pipeline.__call__).parameters:
        call_kwargs['negative_prompt'] = negative
    image = pipeline(**call_kwargs).images[0]

    return image


@_serialized_inference
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
    cancel_event=None,
    on_message=None,
    on_progress=None,
) -> list[Image.Image]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = _get_pipeline(model_path, vae_path, text_encoder_path, on_message=on_message, on_progress=on_progress)

    if seeds is None:
        seeds = list(range(len(prompts)))

    generators = [torch.Generator(device=device).manual_seed(s) for s in seeds]

    _configure_pipeline_loras(pipeline, lora_paths, lora_weights)

    def _step_cb(pipeline, step_index, timestep, callback_kwargs):
        if cancel_event and cancel_event.is_set():
            raise CancelGeneration()
        if progress_cb:
            try:
                progress_cb(step_index, steps)
            except Exception:
                pass
        return callback_kwargs

    negative_prompts = [negative if negative else None] * len(prompts)

    runtime = get_runtime(_detect_model_type(model_path))
    effective_cfg = runtime.effective_cfg(model_path, cfg)

    call_kwargs = dict(
        prompt=prompts,
        num_inference_steps=steps,
        generator=generators,
        width=width,
        height=height,
        guidance_scale=effective_cfg,
        callback_on_step_end=_step_cb,
    )
    if negative and 'negative_prompt' in inspect.signature(pipeline.__call__).parameters:
        call_kwargs['negative_prompt'] = negative_prompts
    result = pipeline(**call_kwargs)

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
    seed: int = 42,
    width: int = 1024,
    height: int = 1024
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
            width=width,
            height=height
        )
        images.append(img)
    return images


_AXIS_APPLIERS = {
    'prompt': lambda p, v: p.update({'prompt': v}),
    'negative': lambda p, v: p.update({'negative': v}),
    'steps': lambda p, v: p.update({'steps': int(v)}),
    'cfg': lambda p, v: p.update({'cfg': float(v)}),
    'model': lambda p, v: p.update({'model_path': v}),
    'vae': lambda p, v: p.update({'vae_path': v}),
    'text_encoder': lambda p, v: p.update({'text_encoder_path': v}),
    'prompt_sr': lambda p, v: p.update({'prompt': v}),
}


def xyz_generate(
    base_params: dict,
    axes: list[tuple[str | None, list]],
    count: int = 1,
    lora_paths: list = None,
    lora_weights: list = None,
    progress_cb=None,
    cancel_event=None,
    on_message=None,
    on_progress=None,
) -> tuple[list[dict], list, list, list]:
    """
    Generate images for X/Y/Z plot (Cartesian product of axes).
    axes: [(type, [values]), ...] — up to 3 (X, Y, Z). type=None means single value.
    base_params: default params to start from before axis overrides.
    Returns (cells, x_vals, y_vals, z_vals) where each cell is:
        {'x': xi, 'y': yi, 'z': zi, 'images': [PIL.Image, ...]}
    """
    import random as _random

    x_type, x_vals = axes[0] if len(axes) > 0 else (None, [None])
    y_type, y_vals = axes[1] if len(axes) > 1 else (None, [None])
    z_type, z_vals = axes[2] if len(axes) > 2 else (None, [None])

    cells = []
    total = len(x_vals) * len(y_vals) * len(z_vals) * count
    done = 0

    def _apply(p, t, v):
        if t and v is not None and t in _AXIS_APPLIERS:
            _AXIS_APPLIERS[t](p, v)

    for xi, xv in enumerate(x_vals):
        for yi, yv in enumerate(y_vals):
            for zi, zv in enumerate(z_vals):
                params = dict(base_params)
                _apply(params, x_type, xv)
                _apply(params, y_type, yv)
                _apply(params, z_type, zv)

                cell_images = []
                for _ in range(count):
                    if cancel_event and cancel_event.is_set():
                        return cells, x_vals, y_vals, z_vals

                    seed = _random.randint(0, 2147483647)
                    img = sdxl_generate(
                        prompt=params.get('prompt', ''),
                        negative=params.get('negative', ''),
                        lora_paths=lora_paths,
                        lora_weights=lora_weights,
                        model_path=params.get('model_path', base_params.get('model_path', 'stabilityai/stable-diffusion-xl-base-1.0')),
                        vae_path=params.get('vae_path'),
                        text_encoder_path=params.get('text_encoder_path'),
                        steps=int(params.get('steps', 20)),
                        cfg=float(params.get('cfg', 7.0)),
                        seed=seed,
                        width=int(params.get('width', 1024)),
                        height=int(params.get('height', 1024)),
                        progress_cb=progress_cb,
                        cancel_event=cancel_event,
                        on_message=on_message,
                        on_progress=on_progress,
                    )
                    cell_images.append(img)
                    done += 1
                    if progress_cb:
                        progress_cb(done, total)

                cells.append({'x': xi, 'y': yi, 'z': zi, 'images': cell_images})

    return cells, x_vals, y_vals, z_vals


def comparison_generate(
    prompts: list[str],
    combos: list[dict],
    negative: str = "",
    steps: int = 20,
    cfg: float = 7.0,
    width: int = 1024,
    height: int = 1024,
    count: int = 1,
    seed: int = 42,
    progress_cb=None,
    cancel_event=None,
) -> tuple[list[dict], list[str]]:
    cells = []
    total = len(prompts) * len(combos) * count
    done = 0

    combo_labels = []
    for c in combos:
        parts = [os.path.basename(c['model_path'])]
        if c.get('lora_paths'):
            weights = c.get('lora_weights') or []
            for index, lp in enumerate(c['lora_paths']):
                weight = weights[index] if index < len(weights) else 1.0
                lora_name = os.path.basename(lp).replace('.safetensors', '')
                parts.append(f"{lora_name} ({float(weight):g})")
        combo_labels.append(c.get('label') or ' + '.join(parts))

    for ci, combo in enumerate(combos):
        for pi, prompt in enumerate(prompts):
            cell_images = []
            cell_seeds = []
            combo_steps = int(combo.get('steps', steps))
            combo_cfg = float(combo.get('cfg', cfg))
            for repeat_index in range(count):
                if cancel_event and cancel_event.is_set():
                    return cells, combo_labels

                # The same prompt/repeat uses the same initial noise for every variant.
                image_seed = seed + (pi * count) + repeat_index
                img = sdxl_generate(
                    prompt=prompt,
                    negative=negative,
                    lora_paths=combo.get('lora_paths'),
                    lora_weights=combo.get('lora_weights'),
                    model_path=combo['model_path'],
                    vae_path=combo.get('vae_path'),
                    text_encoder_path=combo.get('text_encoder_path'),
                    steps=combo_steps,
                    cfg=combo_cfg,
                    seed=image_seed,
                    width=width,
                    height=height,
                    cancel_event=cancel_event,
                    configure_loras=(pi == 0 and repeat_index == 0),
                )
                cell_images.append(img)
                cell_seeds.append(image_seed)
                done += 1
                if progress_cb:
                    progress_cb(done, total)

            cells.append({
                'x': ci,
                'y': pi,
                'images': cell_images,
                'seeds': cell_seeds,
                'steps': combo_steps,
                'cfg': combo_cfg,
            })

    return cells, combo_labels
