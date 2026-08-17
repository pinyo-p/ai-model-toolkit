"""Training recipes and the pinned official Krea 2 trainer adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen


DIFFUSERS_COMMIT = "9284607295a09f759aadd65ed08f48b35feea6d9"
KREA2_TRAINER_SHA256 = "e88d55cb8bf6f1a0d672330661752b982ab7c0a1623980e1673b8b2d303ef948"
KREA2_TRAINER_URL = (
    "https://raw.githubusercontent.com/huggingface/diffusers/"
    f"{DIFFUSERS_COMMIT}/examples/dreambooth/train_dreambooth_lora_krea2.py"
)
PROFILES = {"fast", "balanced", "quality"}
_STEP_PATTERN = re.compile(r"(?:Steps:.*?)(\d+)\s*/\s*(\d+)")
_LOSS_PATTERN = re.compile(r"loss=([0-9.eE+-]+)")


class TrainingError(Exception):
    pass


@dataclass(frozen=True)
class TrainingRecipe:
    engine: str
    profile: str
    training_model: str
    inference_model: str
    instance_prompt: str
    resolution: int
    max_train_steps: int
    rank: int
    lora_alpha: int
    learning_rate: float
    caption_dropout: float
    use_aspect_ratio_buckets: bool
    gradient_checkpointing: bool
    cache_latents: bool
    use_8bit_adam: bool
    lora_layers: str | None
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_trigger(trigger_word: str) -> str:
    return " ".join(trigger_word.strip().split())


def instance_prompt(dataset_type: str, trigger_word: str) -> str:
    trigger = _clean_trigger(trigger_word)
    if not trigger:
        raise TrainingError("Add a trigger word before configuring training")
    templates = {
        "person": f"a photo of {trigger} person",
        "style": trigger,
        "clothing": f"a photo of a person wearing {trigger}",
        "environment": f"a photo of {trigger}",
        "vehicle": f"a photo of {trigger} vehicle",
        "object": f"a photo of {trigger} object",
    }
    if dataset_type not in templates:
        raise TrainingError(f"Unsupported dataset type: {dataset_type}")
    return templates[dataset_type]


def build_krea2_recipe(manifest: dict, profile: str = "balanced", seed: int = 42) -> TrainingRecipe:
    profile = profile.strip().lower()
    if profile not in PROFILES:
        raise TrainingError(f"Unsupported training profile: {profile}")
    seed = int(seed)
    if seed < 0 or seed > 2_147_483_647:
        raise TrainingError("Seed must be between 0 and 2,147,483,647")

    analysis = manifest.get("analysis", {})
    image_count = int(analysis.get("image_count", len(manifest.get("images", []))))
    captioned_count = int(analysis.get("captioned_count", 0))
    if image_count <= 5:
        raise TrainingError("Quick datasets need expansion to more than 5 reviewed images before training")
    if captioned_count != image_count:
        raise TrainingError("Every image needs a caption before training")

    trigger_word = manifest.get("trigger_word", "")
    if not _clean_trigger(trigger_word):
        trigger_word = str(manifest.get("id") or manifest.get("name") or "custom_concept").replace("-", "_")
    prompt = instance_prompt(manifest.get("type", ""), trigger_word)
    if profile == "fast":
        resolution = 768
        max_steps = max(500, min(2000, image_count * 2))
        rank = 16
        learning_rate = 3e-4
        lora_layers = None
    elif profile == "quality":
        resolution = 1024
        max_steps = max(1500, min(6000, image_count * 4))
        rank = 64
        learning_rate = 3e-4
        lora_layers = "to_q,to_k,to_v,to_out.0,to_gate"
    else:
        resolution = 1024
        max_steps = max(1000, min(4000, image_count * 3))
        rank = 32
        learning_rate = 3e-4
        lora_layers = None

    return TrainingRecipe(
        engine="krea2-diffusers",
        profile=profile,
        training_model="krea/Krea-2-Raw",
        inference_model="krea/Krea-2-Turbo",
        instance_prompt=prompt,
        resolution=resolution,
        max_train_steps=max_steps,
        rank=rank,
        lora_alpha=rank,
        learning_rate=learning_rate,
        caption_dropout=0.1 if manifest.get("type") == "style" else 0.0,
        use_aspect_ratio_buckets=True,
        gradient_checkpointing=True,
        cache_latents=True,
        use_8bit_adam=True,
        lora_layers=lora_layers,
        seed=seed,
    )


def _training_caption(manifest: dict, image: dict, prompt: str) -> str:
    caption = " ".join(str(image.get("caption", "")).strip().split())
    trigger = _clean_trigger(manifest.get("trigger_word", ""))
    if manifest.get("type") == "style":
        trigger = trigger or prompt
        return f"{caption}, {trigger}" if caption else trigger
    return f"{prompt}, {caption}" if caption else prompt


def prepare_imagefolder(dataset_dir: str | os.PathLike, manifest: dict, recipe: TrainingRecipe) -> Path:
    images_dir = Path(dataset_dir).resolve() / "images"
    if not images_dir.is_dir():
        raise TrainingError("Dataset image directory is missing")
    records = []
    for image in manifest.get("images", []):
        path = images_dir / image["filename"]
        if not path.is_file():
            raise TrainingError(f"Dataset image is missing: {image['filename']}")
        records.append({
            "file_name": image["filename"],
            "text": _training_caption(manifest, image, recipe.instance_prompt),
        })
    if not records:
        raise TrainingError("Dataset has no images")

    metadata_path = images_dir / "metadata.jsonl"
    temp_path = images_dir / f".metadata.{os.getpid()}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temp_path, metadata_path)
    return images_dir


def ensure_krea2_trainer(project_root: str | os.PathLike, timeout: int = 60) -> Path:
    trainer_dir = Path(project_root).resolve() / "vendor" / "krea2"
    trainer_path = trainer_dir / "train_dreambooth_lora_krea2.py"
    if trainer_path.is_file() and _sha256(trainer_path) == KREA2_TRAINER_SHA256:
        return trainer_path

    trainer_dir.mkdir(parents=True, exist_ok=True)
    request = Request(KREA2_TRAINER_URL, headers={"User-Agent": "ai-toolkit-training-bootstrap/1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
    except Exception as exc:
        raise TrainingError(
            "Could not download the pinned official Krea 2 trainer. Check network access and try again."
        ) from exc
    digest = hashlib.sha256(data).hexdigest()
    if digest != KREA2_TRAINER_SHA256:
        raise TrainingError("Official Krea 2 trainer checksum mismatch; refusing to execute it")

    fd, temporary_name = tempfile.mkstemp(prefix=".trainer-", suffix=".py", dir=trainer_dir)
    try:
        with os.fdopen(fd, "wb") as file:
            file.write(data)
        os.replace(temporary_name, trainer_path)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)
    return trainer_path


def _sha256(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accelerate_executable() -> str:
    executable = Path(sys.executable).resolve().parent / ("accelerate.exe" if os.name == "nt" else "accelerate")
    if executable.is_file():
        return str(executable)
    fallback = shutil.which("accelerate")
    if fallback:
        return fallback
    raise TrainingError("Accelerate CLI is unavailable. Run update.sh to install training dependencies.")


def build_krea2_command(
    trainer_path: str | os.PathLike,
    imagefolder_path: str | os.PathLike,
    output_dir: str | os.PathLike,
    recipe: TrainingRecipe,
) -> list[str]:
    checkpoint_steps = max(100, math.ceil(recipe.max_train_steps / 4))
    command = [
        accelerate_executable(),
        "launch",
        "--num_processes=1",
        str(Path(trainer_path).resolve()),
        "--pretrained_model_name_or_path", recipe.training_model,
        "--dataset_name", str(Path(imagefolder_path).resolve()),
        "--image_column", "image",
        "--caption_column", "text",
        "--output_dir", str(Path(output_dir).resolve()),
        "--instance_prompt", recipe.instance_prompt,
        "--mixed_precision", "bf16",
        "--resolution", str(recipe.resolution),
        "--train_batch_size", "1",
        "--rank", str(recipe.rank),
        "--lora_alpha", str(recipe.lora_alpha),
        "--optimizer", "adamW",
        "--learning_rate", str(recipe.learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(recipe.max_train_steps),
        "--checkpointing_steps", str(checkpoint_steps),
        "--checkpoints_total_limit", "2",
        "--caption_dropout", str(recipe.caption_dropout),
        "--seed", str(recipe.seed),
        "--dataloader_num_workers", "0",
        "--report_to", "tensorboard",
        "--skip_final_inference",
    ]
    if recipe.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if recipe.cache_latents:
        command.append("--cache_latents")
    if recipe.use_8bit_adam:
        command.append("--use_8bit_adam")
    if recipe.use_aspect_ratio_buckets:
        command.append("--use_aspect_ratio_buckets")
    if recipe.lora_layers:
        command.extend(["--lora_layers", recipe.lora_layers])
    return command


def parse_training_progress(line: str) -> dict:
    update = {}
    step_match = _STEP_PATTERN.search(line)
    if step_match:
        completed, total = (int(value) for value in step_match.groups())
        update.update({"completed_steps": completed, "total_steps": total})
    loss_match = _LOSS_PATTERN.search(line)
    if loss_match:
        try:
            update["loss"] = float(loss_match.group(1))
        except ValueError:
            pass
    return update


def find_final_lora(output_dir: str | os.PathLike) -> Path | None:
    output_path = Path(output_dir)
    preferred = output_path / "pytorch_lora_weights.safetensors"
    if preferred.is_file():
        return preferred
    candidates = [path for path in output_path.glob("*.safetensors") if path.is_file()]
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
