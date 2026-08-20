"""Persistent, trainer-agnostic image dataset workspaces."""

from __future__ import annotations

from datetime import datetime, timezone
import copy
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import threading
import uuid
import zipfile

from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError


SCHEMA_VERSION = 2
MAX_IMAGES = 2000
MAX_IMAGE_BYTES = 100 * 1024 * 1024
MAX_IMAGE_PIXELS = 100_000_000
MAX_EXPORT_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_CAPTION_BYTES = 1024 * 1024
MAX_ZIP_MEMBERS = MAX_IMAGES * 3 + 100
QUALITY_AUDIT_VERSION = 1
DATASET_TYPES = {"person", "style", "clothing", "environment", "vehicle", "object"}
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp"}
_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,95}$")
_EVALUATION_IMAGE_PATTERN = re.compile(r"^eval_[a-f0-9-]{32,40}\.png$")
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
    quality_flag_counts = {}
    quality_audited_count = 0
    for item in images:
        width = max(int(item.get("width", 0)), 1)
        height = max(int(item.get("height", 0)), 1)
        ratio = width / height
        bucket = "portrait" if ratio < 0.9 else "landscape" if ratio > 1.1 else "square"
        aspect_counts[bucket] += 1
        quality = item.get("quality")
        if isinstance(quality, dict) and quality.get("version") == QUALITY_AUDIT_VERSION:
            quality_audited_count += 1
            for flag in quality.get("flags", []):
                quality_flag_counts[flag] = quality_flag_counts.get(flag, 0) + 1

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
    quality_flagged_count = sum(
        bool(item.get("quality", {}).get("flags"))
        for item in images
        if isinstance(item.get("quality"), dict)
        and item["quality"].get("version") == QUALITY_AUDIT_VERSION
    )
    if quality_flagged_count:
        issues.append({
            "code": "quality_review",
            "severity": "info",
            "message": f"{quality_flagged_count} images have technical quality flags to review.",
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
        "quality_audited_count": quality_audited_count,
        "quality_flagged_count": quality_flagged_count,
        "quality_flag_counts": quality_flag_counts,
        "issues": issues,
    }


def _histogram_percentile(histogram: list[int], fraction: float) -> int:
    target = max(1, int(sum(histogram) * fraction))
    seen = 0
    for value, count in enumerate(histogram):
        seen += count
        if seen >= target:
            return value
    return 255


def inspect_image_quality(path: str | os.PathLike, width: int, height: int) -> dict:
    """Return conservative technical hints; these are review flags, never blockers."""
    with Image.open(path) as source:
        gray = source.convert("L")
        gray.thumbnail((256, 256), Image.Resampling.LANCZOS)
        brightness = ImageStat.Stat(gray).mean[0]
        histogram = gray.histogram()
        tonal_range = _histogram_percentile(histogram, 0.95) - _histogram_percentile(
            histogram, 0.05
        )
        edges = gray.filter(ImageFilter.FIND_EDGES)
        if edges.width > 2 and edges.height > 2:
            edges = edges.crop((1, 1, edges.width - 1, edges.height - 1))
        sharpness = ImageStat.Stat(edges).var[0]

    flags = []
    if sharpness < 35:
        flags.append("possible_blur")
    if brightness < 18:
        flags.append("very_dark")
    elif brightness > 237:
        flags.append("very_bright")
    if tonal_range < 24:
        flags.append("low_contrast")
    short_edge = max(min(int(width), int(height)), 1)
    long_edge = max(int(width), int(height), 1)
    if long_edge / short_edge > 3:
        flags.append("extreme_aspect")
    return {
        "version": QUALITY_AUDIT_VERSION,
        "audited_at": _utc_now(),
        "sharpness": round(sharpness, 2),
        "mean_brightness": round(brightness, 2),
        "tonal_range": int(tonal_range),
        "flags": flags,
    }


def audit_dataset_quality(
    root: str | os.PathLike,
    dataset_id: str,
    cancel_event: threading.Event | None = None,
    progress_callback=None,
) -> dict:
    manifest_path = _manifest_path(root, dataset_id)
    with _manifest_lock:
        snapshot = _load_manifest(manifest_path)
        images = [dict(item) for item in snapshot.get("images", [])]

    results = {}
    total = len(images)
    for index, item in enumerate(images, start=1):
        if cancel_event and cancel_event.is_set():
            break
        filename = str(item.get("filename", ""))
        if not filename or Path(filename).name != filename:
            raise DatasetError("Dataset image path is invalid")
        path = manifest_path.parent / "images" / filename
        if not path.is_file() or path.is_symlink():
            raise DatasetError(f"Dataset image is missing or invalid: {filename}")
        results[item["id"]] = inspect_image_quality(
            path, int(item.get("width", 0)), int(item.get("height", 0))
        )
        if progress_callback:
            progress_callback(index, total, item.get("original_filename", filename))

    if results:
        with _manifest_lock:
            manifest = _load_manifest(manifest_path)
            for item in manifest.get("images", []):
                quality = results.get(item.get("id"))
                if quality:
                    item["quality"] = quality
            manifest["quality_audited_at"] = _utc_now()
            _save_manifest(manifest_path, manifest)
    else:
        with _manifest_lock:
            manifest = _load_manifest(manifest_path)
    result = _with_analysis(manifest)
    result["quality_audit_result"] = {
        "audited": len(results),
        "total": total,
        "cancelled": bool(cancel_event and cancel_event.is_set()),
    }
    return result


def _with_analysis(manifest: dict) -> dict:
    result = dict(manifest)
    result["dataset_revision"] = _dataset_revision(manifest)
    analysis = analyze_dataset(manifest)
    result["analysis"] = analysis
    completed_runs = [
        run for run in manifest.get("training_runs", [])
        if run.get("status") == "done" and run.get("lora_path")
    ]
    latest_run = completed_runs[-1] if completed_runs else None
    trained_revision = latest_run.get("dataset_revision") if latest_run else None
    if latest_run is None:
        training_status = "none"
    elif not isinstance(trained_revision, int) or trained_revision < 1:
        training_status = "legacy"
    elif trained_revision == result["dataset_revision"]:
        training_status = "current"
    else:
        training_status = "stale"
    result["training_state"] = {
        "status": training_status,
        "current_revision": result["dataset_revision"],
        "trained_revision": trained_revision,
        "current_image_count": len(manifest.get("images", [])),
        "trained_image_count": latest_run.get("dataset_image_count") if latest_run else None,
    }
    if training_status == "stale":
        analysis["issues"].append({
            "code": "stale_training",
            "severity": "warning",
            "message": "This dataset changed after the latest LoRA was trained. Train again when the new revision is ready.",
        })
    elif training_status == "legacy":
        analysis["issues"].append({
            "code": "legacy_training_revision",
            "severity": "info",
            "message": "The latest LoRA predates dataset version tracking, so its exact training revision is unknown.",
        })
    return result


def _dataset_revision(manifest: dict) -> int:
    revision = manifest.get("dataset_revision", 1)
    return revision if isinstance(revision, int) and revision >= 1 else 1


def _advance_dataset_revision(manifest: dict) -> int:
    manifest["dataset_revision"] = _dataset_revision(manifest) + 1
    return manifest["dataset_revision"]


def training_snapshot(manifest: dict) -> dict:
    images = manifest.get("images", [])
    return {
        "dataset_revision": _dataset_revision(manifest),
        "dataset_image_count": len(images),
        "dataset_captioned_count": sum(
            bool(str(item.get("caption", "")).strip()) for item in images
        ),
    }


def _inspect_upload(original_name: str, source: object) -> dict:
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
    source.seek(0)
    return {
        "original_filename": Path(original_name or "image").name[:255],
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
        "bytes": size,
        "sha256": digest.hexdigest(),
    }


def create_dataset(
    root: str | os.PathLike,
    name: str,
    dataset_type: str,
    trigger_word: str,
    uploads: list[tuple[str, object]],
    initial_captions: list[str] | None = None,
    sensitive_content: bool = False,
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
    if initial_captions is not None and len(initial_captions) != len(uploads):
        raise DatasetError("Initial captions must match the uploaded images")

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
        "sensitive_content": sensitive_content is True,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "dataset_revision": 1,
        "images": [],
        "duplicates": [],
    }
    seen_hashes = {}

    try:
        for upload_index, (original_name, source) in enumerate(uploads):
            metadata = _inspect_upload(original_name, source)
            sha256 = metadata["sha256"]
            if sha256 in seen_hashes:
                manifest["duplicates"].append({
                    "original_filename": original_name,
                    "duplicate_of": seen_hashes[sha256],
                    "sha256": sha256,
                })
                continue

            image_id = f"image-{len(manifest['images']) + 1:05d}"
            extension = _FORMAT_EXTENSIONS[metadata["format"]]
            stored_name = f"{image_id}{extension}"
            with open(images_dir / stored_name, "wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)

            caption_value = ""
            if initial_captions is not None:
                caption_value = str(initial_captions[upload_index]).strip()[:4000]
            seen_hashes[sha256] = image_id
            manifest["images"].append({
                "id": image_id,
                "filename": stored_name,
                **metadata,
                "caption": caption_value,
                "caption_source": "import" if caption_value else "none",
            })
            if caption_value:
                with open(images_dir / f"{Path(stored_name).stem}.txt", "w", encoding="utf-8") as file:
                    file.write(caption_value + "\n")

        if not manifest["images"]:
            raise DatasetError("No unique valid images were uploaded")
        _save_manifest(stage_dir / "dataset.json", manifest)
        os.replace(stage_dir, final_dir)
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise

    return _with_analysis(manifest)


def add_images(
    root: str | os.PathLike,
    dataset_id: str,
    uploads: list[tuple[str, object]],
) -> dict:
    if not 1 <= len(uploads) <= MAX_IMAGES:
        raise DatasetError(f"Upload between 1 and {MAX_IMAGES} images")
    manifest_path = _manifest_path(root, dataset_id)
    stage_dir = manifest_path.parent / f".staging-add-{uuid.uuid4().hex}"
    stage_dir.mkdir(parents=True)
    try:
        with _manifest_lock:
            manifest = _load_manifest(manifest_path)
            images = manifest.setdefault("images", [])
            seen_hashes = {
                item.get("sha256"): item.get("id")
                for item in images
                if item.get("sha256")
            }
            image_numbers = [
                int(match.group(1))
                for item in images
                if (match := re.fullmatch(r"image-(\d+)", str(item.get("id", ""))))
            ]
            next_number = max(image_numbers, default=0) + 1
            staged_images = []
            duplicates = []

            for original_name, source in uploads:
                metadata = _inspect_upload(original_name, source)
                sha256 = metadata["sha256"]
                if sha256 in seen_hashes:
                    duplicates.append({
                        "original_filename": original_name,
                        "duplicate_of": seen_hashes[sha256],
                        "sha256": sha256,
                    })
                    continue
                if len(images) + len(staged_images) >= MAX_IMAGES:
                    raise DatasetError(f"Dataset cannot contain more than {MAX_IMAGES} images")

                image_id = f"image-{next_number:05d}"
                next_number += 1
                stored_name = f"{image_id}{_FORMAT_EXTENSIONS[metadata['format']]}"
                with open(stage_dir / stored_name, "wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                seen_hashes[sha256] = image_id
                staged_images.append({
                    "id": image_id,
                    "filename": stored_name,
                    **metadata,
                    "caption": "",
                    "caption_source": "none",
                })

            images_dir = manifest_path.parent / "images"
            images_dir.mkdir(parents=True, exist_ok=True)
            moved_paths = []
            try:
                for item in staged_images:
                    target = images_dir / item["filename"]
                    os.replace(stage_dir / item["filename"], target)
                    moved_paths.append(target)
                images.extend(staged_images)
                if staged_images:
                    _advance_dataset_revision(manifest)
                manifest.setdefault("duplicates", []).extend(duplicates)
                manifest["duplicates"] = manifest["duplicates"][-MAX_IMAGES:]
                _save_manifest(manifest_path, manifest)
            except Exception:
                for path in moved_paths:
                    try:
                        path.unlink()
                    except OSError:
                        pass
                raise
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    result = _with_analysis(manifest)
    result["import_result"] = {
        "added": len(staged_images),
        "duplicates": len(duplicates),
    }
    return result


def remove_images(
    root: str | os.PathLike,
    dataset_id: str,
    image_ids: list[str],
) -> dict:
    selected = {str(image_id) for image_id in image_ids if str(image_id)}
    if not selected:
        raise DatasetError("Select at least one image")
    if len(selected) > MAX_IMAGES:
        raise DatasetError(f"Select at most {MAX_IMAGES} images")

    manifest_path = _manifest_path(root, dataset_id)
    with _manifest_lock:
        manifest = _load_manifest(manifest_path)
        images = manifest.get("images", [])
        known_ids = {item.get("id") for item in images}
        unknown = selected - known_ids
        if unknown:
            raise DatasetError(f"Unknown image id: {sorted(unknown)[0]}")
        if len(selected) >= len(images):
            raise DatasetError("Keep at least one image in the dataset")

        images_dir = manifest_path.parent / "images"
        stage_dir = manifest_path.parent / f".staging-remove-{uuid.uuid4().hex}"
        stage_dir.mkdir(parents=True)
        moved_paths: list[tuple[Path, Path]] = []
        try:
            for item in images:
                if item.get("id") not in selected:
                    continue
                filename = str(item.get("filename", ""))
                if Path(filename).name != filename:
                    raise DatasetError("Dataset image path is invalid")
                image_path = images_dir / filename
                sidecar_path = images_dir / f"{Path(filename).stem}.txt"
                for source in (image_path, sidecar_path):
                    if not source.exists():
                        continue
                    if not source.is_file() or source.is_symlink():
                        raise DatasetError("Dataset image path is invalid")
                    target = stage_dir / source.name
                    os.replace(source, target)
                    moved_paths.append((target, source))

            manifest["images"] = [
                item for item in images if item.get("id") not in selected
            ]
            _advance_dataset_revision(manifest)
            _save_manifest(manifest_path, manifest)
        except Exception:
            for staged, original in reversed(moved_paths):
                try:
                    os.replace(staged, original)
                except OSError:
                    pass
            raise
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    result = _with_analysis(manifest)
    result["remove_result"] = {"removed": len(selected)}
    return result


def update_dataset_settings(
    root: str | os.PathLike,
    dataset_id: str,
    name: str,
    dataset_type: str,
    trigger_word: str,
    sensitive_content: bool | None = None,
) -> dict:
    name = str(name).strip()
    dataset_type = str(dataset_type).strip().lower()
    trigger_word = str(trigger_word).strip()
    if not name:
        raise DatasetError("Dataset name is required")
    if dataset_type not in DATASET_TYPES:
        raise DatasetError(f"Unsupported dataset type: {dataset_type}")

    manifest_path = _manifest_path(root, dataset_id)
    with _manifest_lock:
        manifest = _load_manifest(manifest_path)
        if not trigger_word:
            trigger_word = dataset_id.replace("-", "_")
        new_name = name[:120]
        new_trigger = trigger_word[:120]
        new_sensitive = (
            bool(manifest.get("sensitive_content", False))
            if sensitive_content is None
            else sensitive_content is True
        )
        sensitive_changed = bool(manifest.get("sensitive_content", False)) != new_sensitive
        metadata_changed = manifest.get("name") != new_name or sensitive_changed
        training_changed = (
            manifest.get("type") != dataset_type
            or manifest.get("trigger_word") != new_trigger
        )
        if metadata_changed or training_changed:
            manifest["name"] = new_name
            manifest["type"] = dataset_type
            manifest["trigger_word"] = new_trigger
            manifest["sensitive_content"] = new_sensitive
            if training_changed:
                _advance_dataset_revision(manifest)
            _save_manifest(manifest_path, manifest)

    result = _with_analysis(manifest)
    result["settings_result"] = {
        "changed": metadata_changed or training_changed,
        "training_changed": training_changed,
        "sensitive_changed": sensitive_changed,
    }
    return result


def create_dataset_export(
    root: str | os.PathLike,
    dataset_id: str,
    output_path: str | os.PathLike,
) -> dict:
    manifest_path = _manifest_path(root, dataset_id)
    target = Path(output_path)
    try:
        with _manifest_lock:
            manifest = _load_manifest(manifest_path)
            images_dir = manifest_path.parent / "images"
            metadata_records = []
            portable_images = []
            with zipfile.ZipFile(target, "w", allowZip64=True) as archive:
                for item in manifest.get("images", []):
                    filename = str(item.get("filename", ""))
                    if not filename or Path(filename).name != filename:
                        raise DatasetError("Dataset image path is invalid")
                    source = images_dir / filename
                    if not source.is_file() or source.is_symlink():
                        raise DatasetError(f"Dataset image is missing or invalid: {filename}")
                    archive.write(
                        source,
                        f"images/{filename}",
                        compress_type=zipfile.ZIP_STORED,
                    )
                    caption_value = str(item.get("caption", "")).strip()
                    if caption_value:
                        archive.writestr(
                            f"images/{Path(filename).stem}.txt",
                            caption_value + "\n",
                            compress_type=zipfile.ZIP_DEFLATED,
                        )
                    metadata_records.append({
                        "file_name": f"images/{filename}",
                        "text": caption_value,
                    })
                    portable_images.append({
                        key: item.get(key)
                        for key in (
                            "id", "filename", "original_filename", "width", "height",
                            "format", "mode", "bytes", "sha256", "caption", "caption_source",
                            "generated", "expansion_run_id", "expansion_candidate_id",
                        )
                        if item.get(key) is not None
                    })

                metadata_jsonl = "".join(
                    json.dumps(record, ensure_ascii=False) + "\n"
                    for record in metadata_records
                )
                archive.writestr(
                    "metadata.jsonl", metadata_jsonl, compress_type=zipfile.ZIP_DEFLATED
                )
                export_manifest = {
                    "schema_version": 1,
                    "source_dataset_id": manifest.get("id"),
                    "name": manifest.get("name"),
                    "type": manifest.get("type"),
                    "trigger_word": manifest.get("trigger_word"),
                    "sensitive_content": bool(manifest.get("sensitive_content", False)),
                    "dataset_revision": _dataset_revision(manifest),
                    "created_at": manifest.get("created_at"),
                    "exported_at": _utc_now(),
                    "image_count": len(portable_images),
                    "captioned_count": sum(bool(record["text"]) for record in metadata_records),
                    "images": portable_images,
                }
                archive.writestr(
                    "dataset-export.json",
                    json.dumps(export_manifest, ensure_ascii=False, indent=2),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    return export_manifest


def _read_archive_text(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    label: str,
    max_bytes: int,
) -> str:
    file_type = (info.external_attr >> 16) & 0o170000
    if info.is_dir() or file_type == stat.S_IFLNK or info.flag_bits & 0x1:
        raise DatasetError(f"Dataset ZIP {label} is invalid")
    if info.file_size > max_bytes:
        raise DatasetError(f"Dataset ZIP {label} is too large")
    try:
        return archive.read(info).decode("utf-8")
    except (UnicodeDecodeError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"Dataset ZIP {label} is invalid") from exc


def _import_imagefolder_zip(
    root: str | os.PathLike,
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    name: str,
    dataset_type: str,
    trigger_word: str,
    sensitive_content: bool,
    opened_members: list,
) -> tuple[dict, dict]:
    if not str(name).strip():
        raise DatasetError("Dataset name is required for an ImageFolder ZIP")
    if str(dataset_type).strip().lower() not in DATASET_TYPES:
        raise DatasetError("Select a valid LoRA type for an ImageFolder ZIP")

    image_infos = sorted(
        (
            info for info in infos
            if not info.is_dir()
            and not info.filename.startswith("__MACOSX/")
            and PurePosixPath(info.filename).suffix.lower() in _IMAGE_SUFFIXES
        ),
        key=lambda info: info.filename.casefold(),
    )
    if not 1 <= len(image_infos) <= MAX_IMAGES:
        raise DatasetError(f"ImageFolder ZIP must contain 1-{MAX_IMAGES} images")

    info_by_casefold = {info.filename.casefold(): info for info in infos}
    metadata_captions = {}
    metadata_info = info_by_casefold.get("metadata.jsonl")
    if metadata_info:
        metadata_text = _read_archive_text(
            archive, metadata_info, "metadata.jsonl", MAX_EXPORT_MANIFEST_BYTES
        )
        for line_number, line in enumerate(metadata_text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"Invalid metadata.jsonl line {line_number}") from exc
            if not isinstance(record, dict) or not isinstance(record.get("file_name"), str):
                raise DatasetError(f"Invalid metadata.jsonl line {line_number}")
            metadata_path = PurePosixPath(record["file_name"])
            if (
                metadata_path.is_absolute()
                or ".." in metadata_path.parts
                or "\\" in record["file_name"]
            ):
                raise DatasetError(f"Unsafe metadata.jsonl path on line {line_number}")
            metadata_key = str(metadata_path).casefold()
            if metadata_key in metadata_captions:
                raise DatasetError(f"Duplicate metadata.jsonl path on line {line_number}")
            metadata_captions[metadata_key] = str(record.get("text", ""))

    basename_counts = {}
    for info in image_infos:
        basename = PurePosixPath(info.filename).name.casefold()
        basename_counts[basename] = basename_counts.get(basename, 0) + 1

    uploads = []
    captions = []
    for image_info in image_infos:
        filename = PurePosixPath(image_info.filename).name
        file_type = (image_info.external_attr >> 16) & 0o170000
        if file_type == stat.S_IFLNK or image_info.flag_bits & 0x1:
            raise DatasetError(f"Dataset ZIP image is invalid: {filename}")
        if image_info.file_size <= 0 or image_info.file_size > MAX_IMAGE_BYTES:
            raise DatasetError(f"Dataset ZIP image has an invalid size: {filename}")

        sidecar_name = str(PurePosixPath(image_info.filename).with_suffix(".txt"))
        sidecar_info = info_by_casefold.get(sidecar_name.casefold())
        if sidecar_info:
            caption_value = _read_archive_text(
                archive, sidecar_info, f"caption for {filename}", MAX_CAPTION_BYTES
            ).strip()
        else:
            archive_key = image_info.filename.casefold()
            basename_key = filename.casefold()
            if archive_key in metadata_captions:
                caption_value = metadata_captions[archive_key]
            elif basename_counts[basename_key] == 1 and basename_key in metadata_captions:
                caption_value = metadata_captions[basename_key]
            else:
                caption_value = ""
        try:
            member = archive.open(image_info)
        except (RuntimeError, zipfile.BadZipFile) as exc:
            raise DatasetError(f"Cannot read dataset ZIP image: {filename}") from exc
        opened_members.append(member)
        uploads.append((filename, member))
        captions.append(caption_value)

    result = create_dataset(
        root,
        str(name),
        str(dataset_type),
        str(trigger_word),
        uploads,
        initial_captions=captions,
        sensitive_content=sensitive_content,
    )
    return result, {
        "imported": result["analysis"]["image_count"],
        "source_format": "imagefolder",
        "source_dataset_id": None,
        "source_revision": None,
    }


def import_dataset_archive(
    root: str | os.PathLike,
    archive_source: object,
    name: str = "",
    dataset_type: str = "",
    trigger_word: str = "",
    sensitive_content: bool = False,
) -> dict:
    try:
        archive_source.seek(0)
        archive = zipfile.ZipFile(archive_source)
    except (AttributeError, OSError, zipfile.BadZipFile) as exc:
        raise DatasetError("Invalid dataset ZIP") from exc

    opened_members = []
    try:
        infos = archive.infolist()
        if len(infos) > MAX_ZIP_MEMBERS:
            raise DatasetError(f"Dataset ZIP contains more than {MAX_ZIP_MEMBERS} entries")
        names = [info.filename for info in infos]
        if len(names) != len({name.casefold() for name in names}):
            raise DatasetError("Dataset ZIP contains duplicate paths")
        for member_name in names:
            path = PurePosixPath(member_name)
            if path.is_absolute() or ".." in path.parts or "\\" in member_name:
                raise DatasetError("Dataset ZIP contains an unsafe path")
        for info in infos:
            file_type = (info.external_attr >> 16) & 0o170000
            if file_type == stat.S_IFLNK:
                raise DatasetError("Dataset ZIP contains a symlink")
            if info.flag_bits & 0x1:
                raise DatasetError("Encrypted dataset ZIP files are not supported")

        manifest_info = next(
            (info for info in infos if info.filename == "dataset-export.json"), None
        )
        if manifest_info is None:
            result, import_result = _import_imagefolder_zip(
                root, archive, infos, name, dataset_type, trigger_word,
                sensitive_content, opened_members
            )
            result["import_result"] = import_result
            return result
        if manifest_info.file_size > MAX_EXPORT_MANIFEST_BYTES:
            raise DatasetError("Dataset export manifest is too large")
        try:
            exported = json.loads(archive.read(manifest_info))
        except (json.JSONDecodeError, UnicodeDecodeError, RuntimeError, zipfile.BadZipFile) as exc:
            raise DatasetError("Dataset export manifest is invalid") from exc
        if not isinstance(exported, dict):
            raise DatasetError("Dataset export manifest is invalid")
        if exported.get("schema_version") != 1:
            raise DatasetError("Unsupported dataset export version")
        if not isinstance(exported.get("name"), str) or not exported["name"].strip():
            raise DatasetError("Dataset export name is invalid")
        exported_sensitive = exported.get("sensitive_content", False)
        if not isinstance(exported_sensitive, bool):
            raise DatasetError("Dataset export sensitive-content flag is invalid")

        exported_images = exported.get("images")
        if not isinstance(exported_images, list) or not 1 <= len(exported_images) <= MAX_IMAGES:
            raise DatasetError(f"Dataset ZIP must contain 1-{MAX_IMAGES} images")
        uploads = []
        captions = []
        exported_filenames = set()
        for record in exported_images:
            if not isinstance(record, dict):
                raise DatasetError("Dataset export image metadata is invalid")
            filename = str(record.get("filename", ""))
            if not filename or Path(filename).name != filename:
                raise DatasetError("Dataset export image path is invalid")
            if filename in exported_filenames:
                raise DatasetError("Dataset export contains duplicate image records")
            exported_filenames.add(filename)
            member_name = f"images/{filename}"
            try:
                image_info = archive.getinfo(member_name)
            except KeyError as exc:
                raise DatasetError(f"Dataset ZIP is missing image: {filename}") from exc
            file_type = (image_info.external_attr >> 16) & 0o170000
            if image_info.is_dir() or file_type == stat.S_IFLNK:
                raise DatasetError(f"Dataset ZIP image is invalid: {filename}")
            if image_info.flag_bits & 0x1:
                raise DatasetError("Encrypted dataset ZIP files are not supported")
            if image_info.file_size <= 0 or image_info.file_size > MAX_IMAGE_BYTES:
                raise DatasetError(f"Dataset ZIP image has an invalid size: {filename}")
            try:
                member = archive.open(image_info)
            except (RuntimeError, zipfile.BadZipFile) as exc:
                raise DatasetError(f"Cannot read dataset ZIP image: {filename}") from exc
            opened_members.append(member)
            original_name = Path(str(record.get("original_filename") or filename)).name[:255]
            uploads.append((original_name, member))
            captions.append(str(record.get("caption", "")))

        result = create_dataset(
            root,
            exported["name"],
            str(exported.get("type", "")),
            str(exported.get("trigger_word", "")),
            uploads,
            initial_captions=captions,
            sensitive_content=exported_sensitive,
        )
    except DatasetError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"Cannot import dataset ZIP: {exc}") from exc
    finally:
        for member in opened_members:
            member.close()
        archive.close()

    result["import_result"] = {
        "imported": result["analysis"]["image_count"],
        "source_format": "portable",
        "source_dataset_id": exported.get("source_dataset_id"),
        "source_revision": exported.get("dataset_revision"),
    }
    return result


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
        changed = False
        for item in manifest.get("images", []):
            if item["id"] not in captions:
                continue
            value = str(captions[item["id"]]).strip()[:4000]
            changed = changed or item.get("caption", "") != value
            item["caption"] = value
            item["caption_source"] = source if value else "none"
            sidecar = manifest_path.parent / "images" / f"{Path(item['filename']).stem}.txt"
            if value:
                with open(sidecar, "w", encoding="utf-8") as file:
                    file.write(value + "\n")
            elif sidecar.exists():
                sidecar.unlink()
        if changed:
            _advance_dataset_revision(manifest)
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


def reconcile_interrupted_training_runs(root: str | os.PathLike) -> dict:
    """Mark persisted active runs as interrupted after a server restart."""
    root_path = Path(root).resolve()
    summary = {"datasets_updated": 0, "runs_interrupted": 0}
    if not root_path.exists():
        return summary
    active_statuses = {"queued", "preparing", "running", "cancelling"}
    interrupted_at = _utc_now()
    with _manifest_lock:
        for dataset_dir in root_path.iterdir():
            if not dataset_dir.is_dir() or dataset_dir.name.startswith("."):
                continue
            manifest_path = dataset_dir / "dataset.json"
            try:
                manifest = _load_manifest(manifest_path)
            except DatasetError:
                continue
            changed = 0
            for run in manifest.get("training_runs", []):
                if run.get("status") not in active_statuses:
                    continue
                run["status"] = "interrupted"
                run["interrupted_at"] = interrupted_at
                run["updated_at"] = interrupted_at
                run["message"] = "Server stopped before this training run finished"
                run["error"] = (
                    "Training was interrupted by a server restart; output files were preserved."
                    if run.get("output_dir")
                    else "Training was interrupted by a server restart before an output directory was recorded."
                )
                changed += 1
            if changed:
                _save_manifest(manifest_path, manifest)
                summary["datasets_updated"] += 1
                summary["runs_interrupted"] += changed
    return summary


def record_evaluation_verdict(
    root: str | os.PathLike,
    dataset_id: str,
    training_run_id: str,
    evaluation_id: str,
    experiment: dict,
    comparison: dict,
    votes: dict[int, int],
    summary: dict,
    output_root: str | os.PathLike,
) -> dict:
    """Persist a completed human evaluation against its originating training run."""
    manifest_path = _manifest_path(root, dataset_id)
    with _manifest_lock:
        manifest = _load_manifest(manifest_path)
        run = next(
            (
                item for item in manifest.get("training_runs", [])
                if item.get("job_id") == training_run_id and item.get("status") == "done"
            ),
            None,
        )
        if run is None:
            raise DatasetError("Completed training run not found for this dataset")
        trained_lora = run.get("lora_path")
        if not trained_lora:
            raise DatasetError("Training run does not contain a LoRA file")
        trained_path = Path(trained_lora).resolve()
        tested_paths = {
            Path(lora.get("path", "")).resolve()
            for variant in experiment.get("variants", [])
            for lora in (variant.get("loras") or [])
            if lora.get("path")
        }
        if trained_path not in tested_paths:
            raise DatasetError("Evaluation does not test the LoRA from this training run")

        if not _ID_PATTERN.fullmatch(evaluation_id or ""):
            raise DatasetError("Invalid evaluation id")
        stored_comparison = copy.deepcopy(comparison)
        source_root = Path(output_root).resolve()
        evidence_dir = manifest_path.parent / "evaluations" / evaluation_id
        evidence_dir.mkdir(parents=True, exist_ok=True)
        for cell in stored_comparison.get("cells", []):
            stored_urls = []
            for url in cell.get("images", []):
                filename = Path(url).name
                if (
                    url != f"/output/{filename}"
                    or not _EVALUATION_IMAGE_PATTERN.fullmatch(filename)
                ):
                    raise DatasetError("Evaluation contains an invalid output image")
                source = (source_root / filename).resolve()
                target = evidence_dir / filename
                if source.parent != source_root:
                    raise DatasetError("Evaluation image escapes the output directory")
                if target.is_symlink():
                    raise DatasetError("Evaluation evidence target is invalid")
                if source.is_file():
                    shutil.copy2(source, target)
                elif not target.is_file():
                    raise DatasetError(f"Evaluation image is missing: {filename}")
                stored_urls.append(
                    f"/api/datasets/{dataset_id}/evaluations/{evaluation_id}/images/{filename}"
                )
            cell["images"] = stored_urls

        record = {
            "evaluation_id": evaluation_id,
            "training_run_id": training_run_id,
            "created_at": _utc_now(),
            "votes": {str(key): int(value) for key, value in sorted(votes.items())},
            "summary": summary,
            "experiment": experiment,
            "comparison": stored_comparison,
        }
        evaluations = manifest.setdefault("evaluation_runs", [])
        existing = next(
            (item for item in evaluations if item.get("evaluation_id") == evaluation_id),
            None,
        )
        if existing is None:
            evaluations.append(record)
        else:
            existing.update(record)
        manifest["evaluation_runs"] = evaluations[-20:]
        _save_manifest(manifest_path, manifest)
    return record


def evaluation_image_path(
    root: str | os.PathLike,
    dataset_id: str,
    evaluation_id: str,
    filename: str,
) -> Path:
    if not _ID_PATTERN.fullmatch(evaluation_id or ""):
        raise DatasetNotFound("Evaluation image not found")
    if not _EVALUATION_IMAGE_PATTERN.fullmatch(filename or ""):
        raise DatasetNotFound("Evaluation image not found")
    manifest = get_dataset(root, dataset_id)
    record = next(
        (
            item for item in manifest.get("evaluation_runs", [])
            if item.get("evaluation_id") == evaluation_id
        ),
        None,
    )
    expected_url = (
        f"/api/datasets/{dataset_id}/evaluations/{evaluation_id}/images/{filename}"
    )
    referenced = record and any(
        expected_url in (cell.get("images") or [])
        for cell in (record.get("comparison", {}).get("cells") or [])
    )
    path = _dataset_dir(root, dataset_id) / "evaluations" / evaluation_id / filename
    if not referenced or not path.is_file():
        raise DatasetNotFound("Evaluation image not found")
    return path


def expansion_candidate_dir(
    root: str | os.PathLike, dataset_id: str, run_id: str
) -> Path:
    if not _ID_PATTERN.fullmatch(run_id or ""):
        raise DatasetNotFound("Expansion run not found")
    path = _dataset_dir(root, dataset_id) / "candidates" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def record_expansion_run(root: str | os.PathLike, dataset_id: str, run: dict) -> dict:
    manifest_path = _manifest_path(root, dataset_id)
    with _manifest_lock:
        manifest = _load_manifest(manifest_path)
        runs = manifest.setdefault("expansion_runs", [])
        run_id = run.get("run_id")
        existing = next((item for item in runs if item.get("run_id") == run_id), None)
        if existing is None:
            runs.append(dict(run))
        else:
            existing.update(run)
        manifest["expansion_runs"] = runs[-10:]
        _save_manifest(manifest_path, manifest)
    return _with_analysis(manifest)


def get_expansion_run(root: str | os.PathLike, dataset_id: str, run_id: str) -> dict:
    manifest = get_dataset(root, dataset_id)
    run = next(
        (item for item in manifest.get("expansion_runs", []) if item.get("run_id") == run_id),
        None,
    )
    if run is None:
        raise DatasetNotFound("Expansion run not found")
    return run


def expansion_candidate_path(
    root: str | os.PathLike, dataset_id: str, run_id: str, candidate_id: str
) -> Path:
    if not _ID_PATTERN.fullmatch(candidate_id or ""):
        raise DatasetNotFound("Expansion candidate not found")
    run = get_expansion_run(root, dataset_id, run_id)
    candidate = next(
        (item for item in run.get("candidates", []) if item.get("id") == candidate_id),
        None,
    )
    if candidate is None:
        raise DatasetNotFound("Expansion candidate not found")
    path = expansion_candidate_dir(root, dataset_id, run_id) / candidate["filename"]
    if not path.is_file():
        raise DatasetNotFound("Expansion candidate image not found")
    return path


def review_expansion_candidates(
    root: str | os.PathLike,
    dataset_id: str,
    run_id: str,
    candidate_ids: list[str],
    decision: str,
) -> dict:
    if decision not in {"accept", "reject"}:
        raise DatasetError("Decision must be accept or reject")
    selected = set(candidate_ids)
    if not selected:
        raise DatasetError("Select at least one candidate")

    manifest_path = _manifest_path(root, dataset_id)
    with _manifest_lock:
        manifest = _load_manifest(manifest_path)
        run = next(
            (item for item in manifest.get("expansion_runs", []) if item.get("run_id") == run_id),
            None,
        )
        if run is None:
            raise DatasetNotFound("Expansion run not found")
        candidates = {item.get("id"): item for item in run.get("candidates", [])}
        unknown = selected - set(candidates)
        if unknown:
            raise DatasetNotFound("Expansion candidate not found")

        if decision == "reject":
            for candidate_id in selected:
                candidate = candidates[candidate_id]
                if candidate.get("status") == "pending":
                    candidate["status"] = "rejected"
            _save_manifest(manifest_path, manifest)
            return _with_analysis(manifest)

        existing_hashes = {item.get("sha256") for item in manifest.get("images", [])}
        images_dir = manifest_path.parent / "images"
        accepted_count = 0
        for candidate_id in selected:
            candidate = candidates[candidate_id]
            if candidate.get("status") != "pending":
                continue
            source = manifest_path.parent / "candidates" / run_id / candidate["filename"]
            if not source.is_file():
                raise DatasetNotFound("Expansion candidate image not found")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest in existing_hashes:
                candidate["status"] = "duplicate"
                continue
            with Image.open(source) as image:
                width, height = image.size
                mode = image.mode
            image_id = f"image-{len(manifest['images']) + 1:05d}"
            filename = f"{image_id}.png"
            target = images_dir / filename
            shutil.copy2(source, target)
            caption_value = str(candidate.get("caption", "")).strip()[:4000]
            manifest["images"].append({
                "id": image_id,
                "filename": filename,
                "original_filename": f"generated-{candidate_id}.png",
                "width": width,
                "height": height,
                "format": "PNG",
                "mode": mode,
                "bytes": target.stat().st_size,
                "sha256": digest,
                "caption": caption_value,
                "caption_source": "expansion_recipe" if caption_value else "none",
                "generated": True,
                "expansion_run_id": run_id,
                "expansion_candidate_id": candidate_id,
            })
            if caption_value:
                with open(images_dir / f"{Path(filename).stem}.txt", "w", encoding="utf-8") as file:
                    file.write(caption_value + "\n")
            existing_hashes.add(digest)
            candidate["status"] = "accepted"
            candidate["image_id"] = image_id
            accepted_count += 1

        if accepted_count:
            _advance_dataset_revision(manifest)
        _save_manifest(manifest_path, manifest)
    return _with_analysis(manifest)
