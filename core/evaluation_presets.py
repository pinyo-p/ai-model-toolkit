"""Dataset-aware presets for deterministic LoRA evaluation."""

from __future__ import annotations

from pathlib import Path


class EvaluationPresetError(Exception):
    pass


def _trigger(manifest: dict) -> str:
    value = " ".join(str(manifest.get("trigger_word", "")).strip().split())
    return value or str(manifest.get("id") or "custom_concept").replace("-", "_")


def evaluation_prompts(manifest: dict) -> list[str]:
    trigger = _trigger(manifest)
    prompt_sets = {
        "person": [
            f"a close portrait photo of {trigger} person, soft window light, natural skin detail",
            f"a side-profile portrait of {trigger} person, neutral studio background",
            f"a waist-up photo of {trigger} person outdoors in overcast daylight",
            f"a full-body photo of {trigger} person standing in a minimal studio",
            f"a candid photo of {trigger} person seated in a bright interior",
            f"a photo of {trigger} person under warm evening light",
        ],
        "style": [
            f"a cinematic portrait, {trigger}",
            f"a quiet city street at night, {trigger}",
            f"a mountain landscape at sunrise, {trigger}",
            f"a small reading room with a window, {trigger}",
            f"a futuristic vehicle in a desert, {trigger}",
            f"a still life of fruit and glassware, {trigger}",
        ],
        "clothing": [
            f"a front catalog photo of a person wearing {trigger}",
            f"a back catalog photo of a person wearing {trigger}",
            f"a three-quarter fashion photo of a person wearing {trigger}",
            f"a close detail photo of {trigger}, showing fabric and construction",
            f"a street-style photo of a person wearing {trigger}",
            f"a studio fashion photo of a person wearing {trigger}, dramatic side light",
        ],
        "environment": [
            f"a wide-angle photo of {trigger} in natural daylight",
            f"a view of {trigger} from the opposite corner",
            f"a photo of {trigger} during warm golden hour",
            f"a photo of {trigger} under soft overcast light",
            f"a close architectural detail inside {trigger}",
            f"a centered symmetrical view of {trigger}",
        ],
        "vehicle": [
            f"a front three-quarter photo of {trigger} vehicle on a neutral road",
            f"a rear three-quarter photo of {trigger} vehicle on a neutral road",
            f"a side-profile catalog photo of {trigger} vehicle",
            f"a close photo of the front details of {trigger} vehicle",
            f"a photo of {trigger} vehicle outdoors in overcast daylight",
            f"a cinematic photo of {trigger} vehicle at golden hour",
        ],
        "object": [
            f"a front catalog photo of {trigger} object on a neutral background",
            f"a three-quarter product photo of {trigger} object",
            f"a side view of {trigger} object showing its silhouette",
            f"a macro detail photo of {trigger} object showing its material",
            f"a photo of {trigger} object in an everyday setting",
            f"a studio photo of {trigger} object under soft lighting",
        ],
    }
    try:
        return prompt_sets[manifest["type"]]
    except KeyError as exc:
        raise EvaluationPresetError(f"Unsupported dataset type: {manifest.get('type')}") from exc


def build_evaluation_preset(manifest: dict, run_id: str | None = None) -> dict:
    runs = manifest.get("training_runs", [])
    completed = [run for run in runs if run.get("status") == "done" and run.get("lora_path")]
    if run_id:
        run = next((item for item in completed if item.get("job_id") == run_id), None)
    else:
        run = completed[-1] if completed else None
    if run is None:
        raise EvaluationPresetError("No completed LoRA training run is available for this dataset")
    lora_path = Path(run["lora_path"])
    if not lora_path.is_file():
        raise EvaluationPresetError("The trained LoRA file has been moved or deleted")

    recipe = run.get("recipe") or {}
    model = recipe.get("inference_model") or "krea/Krea-2-Turbo"
    seed = int(recipe.get("seed", run.get("seed", 42)))
    dataset_name = str(manifest.get("name") or "LoRA")
    profile = str(run.get("profile") or "trained")
    return {
        "dataset_id": manifest.get("id"),
        "dataset_name": dataset_name,
        "training_run_id": run.get("job_id"),
        "model": model,
        "steps": 8,
        "cfg": 0.0,
        "seed": seed,
        "width": 1024,
        "height": 1024,
        "images_per_cell": 1,
        "prompts": evaluation_prompts(manifest),
        "variants": [
            {"label": "Base", "lora_path": None, "weight": None},
            {"label": f"{dataset_name} 0.7", "lora_path": str(lora_path), "weight": 0.7},
            {"label": f"{dataset_name} 1.0", "lora_path": str(lora_path), "weight": 1.0},
        ],
        "profile": profile,
    }
