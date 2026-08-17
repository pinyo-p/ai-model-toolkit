"""Persistent, trainer-agnostic image dataset workspaces."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import uuid

from PIL import Image, UnidentifiedImageError


SCHEMA_VERSION = 1
MAX_IMAGES = 2000
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
DATASET_TYPES = {"person", "style", "clothing", "environment", "vehicle", "object"}
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_FORMAT_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "TIFF": ".tiff",
    "BMP": ".bmp",
}
_manifest_lock = threading.RLock()


class DatasetError(Exception):
    pass


class DatasetNotFound(DatasetError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(value: str, fallback: str = "dataset") -> str:
    value = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return (value[:48] or fallback)


def _safe_id(dataset_id: str) -> str:
    if not _ID_PATTERN.fullmatch(dataset_id or ""):
        raise DatasetNotFound("Dataset not found")
    return dataset_id


def _dataset_dir(root: str | os.PathLike, dataset_id: str) -> Path:
    return Path(root).resolve() / _safe_id(dataset_id)


def _manifest_path(root: str | os.PathLike, dataset_id: str) -> Path:
    return _dataset_dir(root, dataset_id) / "dataset.json"


def _save_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = _utc_now()
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def _load_manifest(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as file:
            manifest = json.load(file)
    except FileNotFoundError as exc:
        raise DatasetNotFound("Dataset not found") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read dataset manifest: {exc}") from exc
    return manifest


def analyze_dataset(manifest: dict) -> dict:
    images = manifest.get("images", [])
    image_count = len(images)
    captioned_count = sum(bool(item.get("caption", "").strip()) for item in images)
    low_resolution_count = sum(
        min(int(item.get("width", 0)), int(item.get("height", 0))) < 512
        for item in images
    )
    aspect_counts = {"portrait": 0, "square": 0, "landscape": 0}
    for item in images:
        width = max(int(item.get("width", 0)), 1)
        height = max(int(item.get("height", 0)), 1)
        ratio = width / height
        bucket = "portrait" if ratio < 0.9 else "landscape" if ratio > 1.1 else "square"
        aspect_counts[bucket] += 1

    if image_count <= 5:
        source_mode = "quick"
    elif image_count < 50:
        source_mode = "small"
    else:
        source_mode = "full"

    issues = []
    if captioned_count < image_count:
        issues.append({
            "code": "missing_captions",
            "severity": "warning",
            "message": f"{image_count - captioned_count} images still need captions.",
        })
    if source_mode == "quick":
        issues.append({
            "code": "needs_expansion",
            "severity": "info",
            "message": "Quick dataset: generate and review more views before training.",
        })
    elif source_mode == "small":
        issues.append({
            "code": "small_dataset",
            "severity": "warning",
            "message": "Small dataset: usable for experiments, but it may overfit.",
        })
    if low_resolution_count:
        issues.append({
            "code": "low_resolution",
            "severity": "warning",
            "message": f"{low_resolution_count} images have a short edge below 512 px.",
        })

    if image_count == 0:
        status, next_action = "empty", "Add images"
    elif captioned_count < image_count:
        status, next_action = "needs_captions", "Generate or edit captions"
    elif source_mode == "quick":
        status, next_action = "needs_expansion", "Generate dataset candidates"
    else:
        status, next_action = "ready", "Review and configure training"

    return {
        "status": status,
        "next_action": next_action,
        "source_mode": source_mode,
        "image_count": image_count,
        "captioned_count": captioned_count,
        "duplicate_count": len(manifest.get("duplicates", [])),
        "low_resolution_count": low_resolution_count,
        "aspect_counts": aspect_counts,
        "issues": issues,
    }


def _with_analysis(manifest: dict) -> dict:
    result = dict(manifest)
    result["analysis"] = analyze_dataset(manifest)
    return result


def create_dataset(
    root: str | os.PathLike,
    name: str,
    dataset_type: str,
    trigger_word: str,
    uploads: list[tuple[str, object]],
) -> dict:
    name = name.strip()
    dataset_type = dataset_type.strip().lower()
    trigger_word = trigger_word.strip()
    if not name:
        raise DatasetError("Dataset name is required")
    if dataset_type not in DATASET_TYPES:
        raise DatasetError(f"Unsupported dataset type: {dataset_type}")
    if not 1 <= len(uploads) <= MAX_IMAGES:
        raise DatasetError(f"Upload between 1 and {MAX_IMAGES} images")

    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    dataset_id = f"{_slug(name)}-{uuid.uuid4().hex[:8]}"
    if not trigger_word:
        trigger_word = dataset_id.replace("-", "_")
    stage_dir = root_path / f".staging-{dataset_id}"
    final_dir = root_path / dataset_id
    images_dir = stage_dir / "images"
    images_dir.mkdir(parents=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "id": dataset_id,
        "name": name[:120],
        "type": dataset_type,
        "trigger_word": trigger_word[:120],
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "images": [],
        "duplicates": [],
    }
    seen_hashes = {}

    try:
        for original_name, source in uploads:
            source.seek(0, os.SEEK_END)
            size = source.tell()
            source.seek(0)
            if size <= 0:
                raise DatasetError(f"Empty image: {original_name}")
            if size > MAX_IMAGE_BYTES:
                raise DatasetError(f"Image exceeds 100 MB: {original_name}")

            try:
                with Image.open(source) as image:
                    width, height = image.size
                    image_format = (image.format or "").upper()
                    mode = image.mode
                    if width * height > MAX_IMAGE_PIXELS:
                        raise DatasetError(f"Image is too large: {original_name}")
                    if image_format not in _FORMAT_EXTENSIONS:
                        raise DatasetError(f"Unsupported image format: {original_name}")
                    image.verify()
            except DatasetError:
                raise
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                raise DatasetError(f"Invalid image: {original_name}") from exc

            source.seek(0)
            digest = hashlib.sha256()
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            sha256 = digest.hexdigest()
            if sha256 in seen_hashes:
                manifest["duplicates"].append({
                    "original_filename": original_name,
                    "duplicate_of": seen_hashes[sha256],
                    "sha256": sha256,
                })
                continue

            image_id = f"image-{len(manifest['images']) + 1:05d}"
            extension = _FORMAT_EXTENSIONS[image_format]
            stored_name = f"{image_id}{extension}"
            source.seek(0)
            with open(images_dir / stored_name, "wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)

            seen_hashes[sha256] = image_id
            manifest["images"].append({
                "id": image_id,
                "filename": stored_name,
                "original_filename": Path(original_name or stored_name).name[:255],
                "width": width,
                "height": height,
                "format": image_format,
                "mode": mode,
                "bytes": size,
                "sha256": sha256,
                "caption": "",
                "caption_source": "none",
            })

        if not manifest["images"]:
            raise DatasetError("No unique valid images were uploaded")
        _save_manifest(stage_dir / "dataset.json", manifest)
        os.replace(stage_dir, final_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    return _with_analysis(manifest)


def list_datasets(root: str | os.PathLike) -> list[dict]:
    root_path = Path(root).resolve()
    if not root_path.exists():
        return []
    results = []
    with _manifest_lock:
        for path in root_path.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            try:
                manifest = _load_manifest(path / "dataset.json")
            except DatasetError:
                continue
            summary = _with_analysis(manifest)
            summary.pop("images", None)
            summary.pop("duplicates", None)
            results.append(summary)
    return sorted(results, key=lambda item: item.get("updated_at", ""), reverse=True)


def get_dataset(root: str | os.PathLike, dataset_id: str) -> dict:
    with _manifest_lock:
        return _with_analysis(_load_manifest(_manifest_path(root, dataset_id)))


def image_path(root: str | os.PathLike, dataset_id: str, image_id: str) -> Path:
    manifest = get_dataset(root, dataset_id)
    for item in manifest.get("images", []):
        if item.get("id") == image_id:
            path = _dataset_dir(root, dataset_id) / "images" / item["filename"]
            if path.is_file():
                return path
            break
    raise DatasetNotFound("Dataset image not found")


def update_captions(
    root: str | os.PathLike,
    dataset_id: str,
    captions: dict[str, str],
    source: str = "manual",
) -> dict:
    manifest_path = _manifest_path(root, dataset_id)
    with _manifest_lock:
        manifest = _load_manifest(manifest_path)
        known_ids = {item["id"] for item in manifest.get("images", [])}
        unknown = set(captions) - known_ids
        if unknown:
            raise DatasetError(f"Unknown image id: {sorted(unknown)[0]}")
        for item in manifest.get("images", []):
            if item["id"] not in captions:
                continue
            value = str(captions[item["id"]]).strip()[:4000]
            item["caption"] = value
            item["caption_source"] = source if value else "none"
            sidecar = manifest_path.parent / "images" / f"{Path(item['filename']).stem}.txt"
            if value:
                with open(sidecar, "w", encoding="utf-8") as file:
                    file.write(value + "\n")
            elif sidecar.exists():
                sidecar.unlink()
        _save_manifest(manifest_path, manifest)
    return _with_analysis(manifest)


def record_training_run(root: str | os.PathLike, dataset_id: str, run: dict) -> dict:
    manifest_path = _manifest_path(root, dataset_id)
    with _manifest_lock:
        manifest = _load_manifest(manifest_path)
        runs = manifest.setdefault("training_runs", [])
        run_id = run.get("job_id")
        existing = next((item for item in runs if item.get("job_id") == run_id), None)
        if existing is None:
            runs.append(dict(run))
        else:
            existing.update(run)
        manifest["training_runs"] = runs[-20:]
        _save_manifest(manifest_path, manifest)
    return _with_analysis(manifest)
