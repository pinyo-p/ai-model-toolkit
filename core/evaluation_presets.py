"""Dataset-aware presets for deterministic LoRA evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping


class EvaluationPresetError(Exception):
    pass


def comparison_evidence(result: dict) -> dict:
    """Return a bounded, JSON-safe snapshot of a server-generated comparison grid."""
    if not isinstance(result, dict):
        raise EvaluationPresetError("Evaluation result is missing or invalid")
    experiment = result.get("experiment")
    if not isinstance(experiment, dict):
        raise EvaluationPresetError("Evaluation manifest is missing or invalid")
    prompts = experiment.get("prompts") or []
    variants = experiment.get("variants") or []
    x_labels = result.get("x_labels")
    y_labels = result.get("y_labels")
    cells = result.get("cells")
    if not isinstance(x_labels, list) or len(x_labels) != len(variants):
        raise EvaluationPresetError("Evaluation variant labels do not match its manifest")
    if not isinstance(y_labels, list) or len(y_labels) != len(prompts):
        raise EvaluationPresetError("Evaluation prompt labels do not match its manifest")
    if not isinstance(cells, list) or len(cells) != len(prompts) * len(variants):
        raise EvaluationPresetError("Evaluation cells do not match its manifest")

    saved_cells = []
    image_count = 0
    positions = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise EvaluationPresetError("Evaluation contains an invalid cell")
        try:
            x = int(cell.get("x"))
            y = int(cell.get("y"))
        except (TypeError, ValueError) as exc:
            raise EvaluationPresetError("Evaluation contains an invalid cell position") from exc
        if x < 0 or x >= len(variants) or y < 0 or y >= len(prompts):
            raise EvaluationPresetError("Evaluation contains an unknown cell position")
        if (x, y) in positions:
            raise EvaluationPresetError("Evaluation contains a duplicate cell position")
        positions.add((x, y))
        images = cell.get("images") or []
        if not isinstance(images, list) or not images or any(
            not isinstance(url, str) or not url.startswith("/output/") for url in images
        ):
            raise EvaluationPresetError("Evaluation contains an invalid image reference")
        try:
            seeds = [int(seed) for seed in (cell.get("seeds") or [])]
        except (TypeError, ValueError) as exc:
            raise EvaluationPresetError("Evaluation contains an invalid seed") from exc
        if len(seeds) != len(images):
            raise EvaluationPresetError("Evaluation image and seed counts do not match")
        image_count += len(images)
        if image_count > 500:
            raise EvaluationPresetError("Evaluation evidence exceeds 500 images")
        saved_cells.append({
            "x": x,
            "y": y,
            "images": images,
            "seeds": seeds,
            "steps": cell.get("steps"),
            "cfg": cell.get("cfg"),
        })
    return {
        "x_labels": [str(label) for label in x_labels],
        "y_labels": [str(label) for label in y_labels],
        "cells": saved_cells,
        "image_count": image_count,
    }


def summarize_votes(
    experiment: dict,
    votes: Mapping[str | int, int],
    *,
    require_complete: bool = False,
) -> dict:
    """Validate human picks and return a deterministic evaluation summary."""
    if not isinstance(experiment, dict):
        raise EvaluationPresetError("Evaluation manifest is missing or invalid")
    prompts = experiment.get("prompts")
    variants = experiment.get("variants")
    if not isinstance(prompts, list) or not prompts or len(prompts) > 100:
        raise EvaluationPresetError("Evaluation must contain between 1 and 100 prompts")
    if not isinstance(variants, list) or not variants or len(variants) > 100:
        raise EvaluationPresetError("Evaluation must contain between 1 and 100 variants")
    if not isinstance(votes, Mapping):
        raise EvaluationPresetError("Votes must map prompt rows to winning variants")

    normalized: dict[int, int] = {}
    for raw_prompt, raw_variant in votes.items():
        try:
            prompt_index = int(raw_prompt)
            variant_index = int(raw_variant)
        except (TypeError, ValueError) as exc:
            raise EvaluationPresetError("Vote indices must be integers") from exc
        if str(prompt_index) != str(raw_prompt) and raw_prompt != prompt_index:
            raise EvaluationPresetError("Vote prompt indices must be canonical integers")
        if prompt_index < 0 or prompt_index >= len(prompts):
            raise EvaluationPresetError("Vote refers to an unknown prompt row")
        if variant_index < 0 or variant_index >= len(variants):
            raise EvaluationPresetError("Vote refers to an unknown variant")
        normalized[prompt_index] = variant_index

    if require_complete and len(normalized) != len(prompts):
        raise EvaluationPresetError(
            f"Choose a winner for every prompt ({len(normalized)}/{len(prompts)} reviewed)"
        )

    counts = [0] * len(variants)
    for variant_index in normalized.values():
        counts[variant_index] += 1
    highest = max(counts) if normalized else 0
    leaders = [index for index, count in enumerate(counts) if count == highest] if normalized else []
    winner_index = leaders[0] if len(leaders) == 1 else None
    winner = None
    verdict = "inconclusive"
    if winner_index is not None:
        variant = variants[winner_index]
        winner = {
            "index": winner_index,
            "label": str(variant.get("label") or f"Variant {winner_index + 1}"),
            "votes": counts[winner_index],
        }
        loras = variant.get("loras") or []
        verdict = "lora" if loras else "base"
        if len(loras) == 1:
            winner["lora_weight"] = loras[0].get("weight")

    return {
        "prompt_count": len(prompts),
        "reviewed_count": len(normalized),
        "counts": [
            {
                "index": index,
                "label": str(variant.get("label") or f"Variant {index + 1}"),
                "votes": counts[index],
            }
            for index, variant in enumerate(variants)
        ],
        "winner": winner,
        "tied_variant_indices": leaders if len(leaders) > 1 else [],
        "verdict": verdict,
    }


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
