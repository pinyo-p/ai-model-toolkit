"""Reference-image dataset expansion with simple concept-aware recipes."""

from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from pathlib import Path
import threading

from PIL import Image
import torch

from . import sdxl


MODEL_ID = "Qwen/Qwen-Image-Edit-2509"
INFERENCE_STEPS = 50
MAX_REFERENCE_IMAGES = 3


class ExpansionError(Exception):
    pass


class ExpansionCancelled(ExpansionError):
    pass


@dataclass(frozen=True)
class Variation:
    instruction: str
    caption: str


_VARIATIONS = {
    "person": [
        ("a close portrait from a three-quarter angle in soft window light", "a close three-quarter portrait in soft window light"),
        ("a clean side-profile portrait against a simple neutral background", "a side-profile portrait against a neutral background"),
        ("a waist-up portrait outdoors in open shade", "a waist-up portrait outdoors in soft daylight"),
        ("a full-body standing portrait in a minimal studio", "a full-body standing portrait in a minimal studio"),
        ("a seated candid portrait in a bright interior", "a seated candid portrait in a bright interior"),
        ("a walking full-body view on a quiet street", "a full-body walking view on a quiet street"),
        ("a close portrait under warm evening light", "a close portrait under warm evening light"),
        ("a waist-up portrait under cool overcast daylight", "a waist-up portrait under cool overcast daylight"),
    ],
    "clothing": [
        ("a clean front view worn by a model against a neutral studio background", "a front view of the clothing worn in a neutral studio"),
        ("a clean back view worn by a model against a neutral studio background", "a back view of the clothing worn in a neutral studio"),
        ("a three-quarter view showing the garment shape and fit", "a three-quarter view showing the clothing fit"),
        ("a close detail view of the fabric, seams, and distinctive features", "a close detail of the clothing fabric and construction"),
        ("the same garment laid flat in a clean catalog photograph", "the clothing laid flat in a clean catalog photograph"),
        ("the same garment on a mannequin in soft daylight", "the clothing displayed on a mannequin in soft daylight"),
    ],
    "environment": [
        ("the same place viewed from a wider angle in natural daylight", "a wide view of the place in natural daylight"),
        ("the same place viewed from the opposite corner", "the place viewed from the opposite corner"),
        ("the same place at warm golden hour", "the place in warm golden-hour light"),
        ("the same place under soft overcast light", "the place under soft overcast light"),
        ("a closer view emphasizing the characteristic materials and details", "a close view of the characteristic materials and details"),
        ("a centered architectural view with clean perspective", "a centered view of the place with clean perspective"),
    ],
    "vehicle": [
        ("a front three-quarter exterior view on a neutral road", "a front three-quarter view of the vehicle"),
        ("a rear three-quarter exterior view on a neutral road", "a rear three-quarter view of the vehicle"),
        ("a clean side-profile catalog view", "a side-profile view of the vehicle"),
        ("a close view of the front design details", "a close view of the vehicle front details"),
        ("a close view of the rear design details", "a close view of the vehicle rear details"),
        ("the vehicle outdoors under soft overcast daylight", "the vehicle outdoors under soft overcast daylight"),
    ],
    "object": [
        ("a clean front catalog view against a neutral background", "a front catalog view of the object"),
        ("a three-quarter product view against a neutral background", "a three-quarter product view of the object"),
        ("a side view showing the complete silhouette", "a side view of the object"),
        ("a close detail view showing material and construction", "a close detail of the object's material and construction"),
        ("the object in a natural everyday setting", "the object in a natural everyday setting"),
        ("the object under soft studio lighting", "the object under soft studio lighting"),
    ],
    "style": [
        ("change the subject to a quiet city street while preserving only the exact visual style", "a quiet city street"),
        ("change the subject to a portrait while preserving only the exact visual style", "a portrait"),
        ("change the subject to a mountain landscape while preserving only the exact visual style", "a mountain landscape"),
        ("change the subject to a small interior room while preserving only the exact visual style", "a small interior room"),
        ("change the subject to a vehicle while preserving only the exact visual style", "a vehicle"),
        ("change the subject to a still life while preserving only the exact visual style", "a still life"),
    ],
}


_pipeline = None
_pipeline_lock = threading.RLock()


def build_variations(dataset_type: str, count: int) -> list[Variation]:
    if dataset_type not in _VARIATIONS:
        raise ExpansionError(f"Unsupported dataset type: {dataset_type}")
    if count < 1 or count > 24:
        raise ExpansionError("Generate between 1 and 24 candidates per run")
    preserve = {
        "person": "Preserve the exact identity, facial features, apparent age, hair, and body proportions from the reference image(s). Create ",
        "clothing": "Preserve the exact garment design, colors, pattern, materials, and logos from the reference image(s). Create ",
        "environment": "Preserve the identity, layout, architecture, and distinctive features of the place from the reference image(s). Create ",
        "vehicle": "Preserve the exact vehicle identity, shape, paint, trim, and distinctive details from the reference image(s). Create ",
        "object": "Preserve the exact object identity, shape, colors, materials, and distinctive details from the reference image(s). Create ",
        "style": "Use the reference image(s) only as a visual-style reference. Do not copy their subject. Precisely preserve palette, medium, texture, rendering, and composition language; ",
    }[dataset_type]
    suffix = (
        ". Keep the result realistic and internally consistent. Do not add text, watermarks, borders, "
        "collages, duplicate subjects, or comparison panels. Produce one image."
    )
    templates = _VARIATIONS[dataset_type]
    return [
        Variation(instruction=preserve + templates[index % len(templates)][0] + suffix, caption=templates[index % len(templates)][1])
        for index in range(count)
    ]


def _get_pipeline():
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        if not torch.cuda.is_available():
            raise ExpansionError("Dataset expansion requires an NVIDIA CUDA GPU")
        try:
            from diffusers import QwenImageEditPlusPipeline
        except ImportError as exc:
            raise ExpansionError(
                "Qwen Image Edit Plus is unavailable. Run update.sh to install the compatible Diffusers build."
            ) from exc
        sdxl.release_cached_pipelines()
        kwargs = {"torch_dtype": torch.bfloat16}
        token = os.environ.get("HF_TOKEN")
        if token:
            kwargs["token"] = token
        _pipeline = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, **kwargs).to("cuda")
        return _pipeline


def release_pipeline():
    """Free the editor after a batch so Krea 2 training has the full GPU budget."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_candidate(
    reference_paths: list[str | os.PathLike],
    instruction: str,
    seed: int,
    progress_cb=None,
    cancel_event=None,
) -> Image.Image:
    if not reference_paths:
        raise ExpansionError("At least one reference image is required")
    references = []
    for path in reference_paths[:MAX_REFERENCE_IMAGES]:
        with Image.open(Path(path)) as image:
            references.append(image.convert("RGB"))

    with sdxl.inference_session():
        pipeline = _get_pipeline()
        generator = torch.Generator(device="cuda").manual_seed(int(seed))

        def step_callback(pipe, step_index, timestep, callback_kwargs):
            if cancel_event and cancel_event.is_set():
                raise ExpansionCancelled("Dataset expansion cancelled")
            if progress_cb:
                progress_cb(step_index + 1, INFERENCE_STEPS)
            return callback_kwargs

        result = pipeline(
            image=references if len(references) > 1 else references[0],
            prompt=instruction,
            negative_prompt="low quality, blurry, distorted, duplicate, collage, text, watermark",
            num_inference_steps=INFERENCE_STEPS,
            true_cfg_scale=4.0,
            generator=generator,
            callback_on_step_end=step_callback,
        )
    return result.images[0]
