"""Model-family runtime registry for image generation pipelines.

Each runtime owns family-specific loading and inference defaults. The generation
orchestrator in ``core.sdxl`` remains responsible for caching, LoRA lifecycle,
seeds, progress callbacks, and invoking the selected pipeline.
"""

from dataclasses import dataclass
import json
import os
import struct
from typing import Callable

import torch
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)
from fastapi import HTTPException

from .flux2 import load_base_flux2_and_swap_weights
from .zimage import load_zimage_pipeline


@dataclass(frozen=True)
class RuntimeDefaults:
    steps: int
    cfg: float


@dataclass
class RuntimeLoadContext:
    model_path: str
    vae_path: str | None
    text_encoder_path: str | None
    vae: object | None
    dtype: torch.dtype
    load_pipeline: Callable
    on_message: Callable | None = None
    on_progress: Callable | None = None


class ModelRuntime:
    family = "sdxl"
    dtype = torch.float16

    def defaults(self, model_path: str) -> RuntimeDefaults:
        return RuntimeDefaults(steps=20, cfg=7.0)

    def effective_cfg(self, model_path: str, requested_cfg: float) -> float:
        return requested_cfg

    def load(self, context: RuntimeLoadContext):
        raise NotImplementedError


class SDXLRuntime(ModelRuntime):
    family = "sdxl"

    def load(self, context: RuntimeLoadContext):
        vae = context.vae
        if vae is None:
            try:
                vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sdxl-vae", torch_dtype=context.dtype
                )
            except Exception:
                pass
        return context.load_pipeline(
            StableDiffusionXLPipeline,
            context.model_path,
            vae=vae,
            dtype=context.dtype,
        )


class SD15Runtime(ModelRuntime):
    family = "sd15"

    def load(self, context: RuntimeLoadContext):
        return context.load_pipeline(
            StableDiffusionPipeline, context.model_path, dtype=context.dtype
        )


class ZImageRuntime(ModelRuntime):
    family = "zimage"
    dtype = torch.bfloat16

    def defaults(self, model_path: str) -> RuntimeDefaults:
        return RuntimeDefaults(steps=9, cfg=0.0)

    def load(self, context: RuntimeLoadContext):
        return load_zimage_pipeline(
            context.model_path,
            context.dtype,
            vae_path=context.vae_path,
            text_encoder_path=context.text_encoder_path,
            local_vae=context.vae,
            on_message=context.on_message,
        )


class PixArtRuntime(ModelRuntime):
    family = "pixart"

    def defaults(self, model_path: str) -> RuntimeDefaults:
        return RuntimeDefaults(steps=20, cfg=5.0)

    def load(self, context: RuntimeLoadContext):
        try:
            from diffusers import PixArtAlphaPipeline

            return context.load_pipeline(
                PixArtAlphaPipeline, context.model_path, dtype=context.dtype
            )
        except Exception:
            kwargs = {"vae": context.vae, "dtype": context.dtype}
            if context.text_encoder_path and os.path.exists(context.text_encoder_path):
                try:
                    from transformers import CLIPTextModel

                    kwargs["text_encoder"] = CLIPTextModel.from_pretrained(
                        context.text_encoder_path, torch_dtype=context.dtype
                    )
                except Exception:
                    pass
            return context.load_pipeline(
                StableDiffusionXLPipeline, context.model_path, **kwargs
            )


class FluxRuntime(ModelRuntime):
    family = "flux"

    def defaults(self, model_path: str) -> RuntimeDefaults:
        return RuntimeDefaults(steps=28, cfg=3.5)

    def load(self, context: RuntimeLoadContext):
        try:
            from diffusers import FluxPipeline

            return context.load_pipeline(
                FluxPipeline, context.model_path, dtype=context.dtype
            )
        except Exception as exc:
            if any(marker in str(exc) for marker in ("Mistral", "text_model", "Qwen")):
                if os.path.isdir(context.model_path):
                    return context.load_pipeline(
                        DiffusionPipeline, context.model_path, dtype=context.dtype
                    )
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "This appears to be a FLUX.2 single file, which is not supported.\n"
                        "Use the full directory format instead."
                    ),
                )

            kwargs = {"vae": context.vae, "dtype": context.dtype}
            if context.text_encoder_path and os.path.exists(context.text_encoder_path):
                try:
                    from transformers import CLIPTextModel

                    kwargs["text_encoder"] = CLIPTextModel.from_pretrained(
                        context.text_encoder_path, torch_dtype=context.dtype
                    )
                except Exception:
                    pass
            return context.load_pipeline(
                StableDiffusionXLPipeline, context.model_path, **kwargs
            )


class Flux2Runtime(ModelRuntime):
    family = "flux2"
    dtype = torch.bfloat16

    def defaults(self, model_path: str) -> RuntimeDefaults:
        name = os.path.basename(model_path).lower()
        if "klein" in name or "schnell" in name:
            return RuntimeDefaults(steps=4, cfg=1.0)
        return RuntimeDefaults(steps=28, cfg=4.0)

    def effective_cfg(self, model_path: str, requested_cfg: float) -> float:
        name = os.path.basename(model_path).lower()
        return 1.0 if "klein" in name or "schnell" in name else requested_cfg

    def load(self, context: RuntimeLoadContext):
        if os.path.isfile(context.model_path):
            return load_base_flux2_and_swap_weights(
                context.model_path,
                context.dtype,
                os.environ.get("HF_TOKEN"),
                on_message=context.on_message,
                on_progress=context.on_progress,
            )
        return context.load_pipeline(
            DiffusionPipeline, context.model_path, dtype=context.dtype
        )


class Krea2Runtime(ModelRuntime):
    family = "krea2"
    dtype = torch.bfloat16

    def defaults(self, model_path: str) -> RuntimeDefaults:
        if "turbo" in model_path.lower():
            return RuntimeDefaults(steps=8, cfg=0.0)
        # Official Krea 2 Raw release recipe.
        return RuntimeDefaults(steps=52, cfg=3.5)

    def effective_cfg(self, model_path: str, requested_cfg: float) -> float:
        return 0.0 if "turbo" in model_path.lower() else requested_cfg

    def load(self, context: RuntimeLoadContext):
        if os.path.isfile(context.model_path):
            raise RuntimeError(
                "Krea 2 single-file checkpoints need the official standalone runtime. "
                "For AI Toolkit, download the full Diffusers repository "
                "(krea/Krea-2-Raw or krea/Krea-2-Turbo)."
            )
        try:
            from diffusers import Krea2Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Krea 2 requires a Diffusers build with Krea2Pipeline. "
                "Run update.sh to install the pinned compatible build."
            ) from exc

        return context.load_pipeline(
            Krea2Pipeline, context.model_path, dtype=context.dtype
        )


_RUNTIMES = {
    "sdxl": SDXLRuntime(),
    "sd15": SD15Runtime(),
    "zimage": ZImageRuntime(),
    "pixart": PixArtRuntime(),
    "flux": FluxRuntime(),
    "flux2": Flux2Runtime(),
    "krea2": Krea2Runtime(),
}


def get_runtime(family: str) -> ModelRuntime:
    return _RUNTIMES.get(family, _RUNTIMES["sdxl"])


def get_runtime_defaults(model_path: str, family: str | None = None) -> RuntimeDefaults:
    resolved_family = family or detect_model_type(model_path)
    return get_runtime(resolved_family).defaults(model_path)


def _read_safetensors_keys(path: str):
    try:
        with open(path, "rb") as file:
            header_length = struct.unpack("<Q", file.read(8))[0]
            if header_length <= 0 or header_length > 50 * 1024 * 1024:
                return None
            raw = file.read(header_length)
            if len(raw) != header_length:
                return None
            header = json.loads(raw)
        return [key for key in header if key != "__metadata__"]
    except Exception:
        return None


def detect_model_type(model_path: str) -> str:
    model_lower = model_path.lower()
    if any(marker in model_lower for marker in ("krea-2", "krea_2", "krea2")):
        return "krea2"
    if any(marker in model_lower for marker in ("z-image", "z_image", "zimage")):
        return "zimage"
    if any(marker in model_lower for marker in ("flux2", "flux.2", "flux-2")):
        return "flux2"
    if "flux" in model_lower:
        return "flux"
    if any(marker in model_lower for marker in ("xl", "sdxl", "pony", "sd_xl", "illustrious")):
        return "sdxl"
    if any(marker in model_lower for marker in ("v1-5", "v1.5", "sd15", "sd-1", "runwayml")):
        return "sd15"

    if model_path.endswith(".safetensors") and os.path.isfile(model_path):
        filename = os.path.basename(model_path).lower()
        keys = _read_safetensors_keys(model_path)
        if keys:
            joined = " ".join(key.lower() for key in keys)
            if "single_stream_blocks" in joined and "double_stream" not in joined:
                return "zimage"
            if any(marker in joined for marker in ("noise_refiner", "cap_embedder", "context_refiner")):
                return "zimage"
            if "mmdit." in joined:
                return "sd3"
            if "model.diffusion_model" in joined:
                has_double = "double_blocks" in joined
                has_single = "single_blocks" in joined
                if has_double and has_single:
                    return "flux2"
                if has_double:
                    return "flux"
                if any(marker in joined for marker in ("input_blocks.", "mid_block.", "output_blocks.")):
                    return "sdxl"
                if "x_embedder" in joined and "model.diffusion_model.layers." in joined:
                    return "pixart"
                return "sdxl"
            if "double_stream" in joined:
                return "flux2" if any(marker in filename for marker in ("flux2", "flux.2", "flux-2")) else "flux"
            if "transformer_blocks" in joined and "time_text_embed" in joined:
                return "flux"
            if "transformer_blocks" in joined and ("attn1" in joined or "attn2" in joined):
                return "pixart"
            if "x_embedder" in joined and "layers." in joined:
                return "pixart"

    if os.path.isdir(model_path):
        index_path = os.path.join(model_path, "model_index.json")
        if os.path.exists(index_path):
            try:
                with open(index_path) as file:
                    class_name = json.load(file).get("_class_name", "")
                mapping = {
                    "StableDiffusionPipeline": "sd15",
                    "StableDiffusionXLPipeline": "sdxl",
                    "StableDiffusion3Pipeline": "sd3",
                    "FluxPipeline": "flux",
                    "Flux2Pipeline": "flux2",
                    "Flux2KleinPipeline": "flux2",
                    "Flux2KleinKVPipeline": "flux2",
                    "ZImagePipeline": "zimage",
                    "Krea2Pipeline": "krea2",
                    "HunyuanDiTPipeline": "hunyuan",
                    "PixArtAlphaPipeline": "pixart",
                    "KolorsPipeline": "kolors",
                }
                return mapping.get(class_name, "sdxl")
            except Exception:
                pass

    return "sdxl"
