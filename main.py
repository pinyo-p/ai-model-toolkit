import os
import sys
import struct
import secrets
import datetime
import shutil
import tempfile
import uuid
import json
import io
import sqlite3
import hashlib
import subprocess
import signal
from typing import List

import torch
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
from pydantic import BaseModel
from contextlib import asynccontextmanager
from starlette.background import BackgroundTask

from core import datasets as dataset_module
from core import gpu, caption, sdxl, lora, training, expansion, evaluation_presets, image as img_module, utils
import threading
import time

security = HTTPBasic()

_download_progress: dict = {}
_dl_lock = threading.Lock()

_generate_progress: dict = {}
_gen_lock = threading.Lock()
_gen_cancel_events: dict = {}
_gen_cancel_lock = threading.Lock()

_evaluation_progress: dict = {}
_evaluation_lock = threading.Lock()
_evaluation_cancel_events: dict = {}
_evaluation_cancel_lock = threading.Lock()

_dataset_caption_jobs: dict = {}
_dataset_caption_lock = threading.Lock()
_dataset_caption_cancel_events: dict = {}

_training_jobs: dict = {}
_training_lock = threading.Lock()
_training_processes: dict = {}
_training_cancel_events: dict = {}

_expansion_jobs: dict = {}
_expansion_lock = threading.Lock()
_expansion_cancel_events: dict = {}

_dataset_exports: dict = {}
_dataset_export_lock = threading.Lock()
_DATASET_EXPORT_TTL_SECONDS = 10 * 60


def _dataset_has_active_job(dataset_id: str) -> bool:
    with _dataset_caption_lock:
        if any(
            job.get("dataset_id") == dataset_id
            and job.get("status") in {"queued", "running"}
            for job in _dataset_caption_jobs.values()
        ):
            return True
    with _expansion_lock:
        if any(
            job.get("dataset_id") == dataset_id
            and job.get("status") in {"queued", "loading", "running", "cancelling"}
            for job in _expansion_jobs.values()
        ):
            return True
    with _training_lock:
        return any(
            job.get("dataset_id") == dataset_id
            and job.get("status") in {"queued", "preparing", "running", "cancelling"}
            for job in _training_jobs.values()
        )


def _unlink_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _prune_dataset_exports(now: float) -> None:
    expired = [
        token for token, item in _dataset_exports.items()
        if now - item["created_at"] > _DATASET_EXPORT_TTL_SECONDS
    ]
    for token in expired:
        item = _dataset_exports.pop(token)
        _unlink_quietly(item["path"])


def _expire_dataset_export(token: str) -> None:
    with _dataset_export_lock:
        item = _dataset_exports.pop(token, None)
    if item:
        _unlink_quietly(item["path"])


DB_FILE = os.path.join(os.path.dirname(__file__), "users.db")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user'
        )
    """)
    admin = conn.execute("SELECT * FROM users WHERE username = 'admin'").fetchone()
    if not admin:
        hashed = hashlib.sha256("admin".encode()).hexdigest()
        conn.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)", ("admin", hashed, "admin"))
    conn.commit()
    conn.close()

init_db()


def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    conn = sqlite3.connect(DB_FILE)
    user = conn.execute("SELECT * FROM users WHERE username = ?", (credentials.username,)).fetchone()
    conn.close()
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    hashed = hashlib.sha256(credentials.password.encode()).hexdigest()
    if user[1] != hashed:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return user[0]


temp_dir = tempfile.mkdtemp()

SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {
        "hf_token": "",
        "civitai_token": "",
        "models_path": os.path.join(os.path.expanduser("~"), "models"),
        "base_url": "",
        "api_keys": [],
        "public_downloads": []
    }

def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)

settings = load_settings()


def _datasets_root() -> str:
    return settings.get("datasets_path") or os.path.join(os.path.dirname(__file__), "datasets")

# Set HF token from settings as env var so core/sdxl.py can use it
if settings.get("hf_token"):
    os.environ["HF_TOKEN"] = settings["hf_token"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    gpu_info = gpu.check_gpu()
    print(f"👑 ai-toolkit ready on GB10 | GPU: {gpu_info['gpu_name']} | VRAM: {gpu_info['vram_total_gb']}GB")
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Output directory for generated images (served without auth)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api/gpu")
async def api_gpu():
    return gpu.check_gpu()


@app.post("/api/caption")
async def api_caption(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    try:
        image_data = await file.read()
        image_path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
        with open(image_path, "wb") as f:
            f.write(image_data)

        result = caption.image_captioning(image_path)
        os.remove(image_path)

        return {"caption": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/auto_caption")
async def api_auto_caption(files: list[UploadFile] = File(...), user: str = Depends(get_current_user)):
    try:
        if len(files) < 1:
            raise HTTPException(status_code=400, detail="Upload at least 1 image")

        image_paths = []
        for f in files:
            data = await f.read()
            path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
            with open(path, "wb") as fp:
                fp.write(data)
            image_paths.append(path)

        results = caption.auto_caption(image_paths)

        for p in image_paths:
            os.remove(p)

        return {"captions": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DatasetCaptionPayload(BaseModel):
    captions: dict[str, str]


class DatasetImageRemovePayload(BaseModel):
    image_ids: list[str]


class DatasetSettingsPayload(BaseModel):
    name: str
    dataset_type: str
    trigger_word: str = ""
    sensitive_content: bool | None = None


class TrainingStartPayload(BaseModel):
    profile: str = "balanced"
    seed: int = 42


class ExpansionStartPayload(BaseModel):
    candidate_count: int = 8
    seed: int = 42


class ExpansionReviewPayload(BaseModel):
    candidate_ids: list[str]
    decision: str


class EvaluationVerdictPayload(BaseModel):
    training_run_id: str
    evaluation_id: str
    votes: dict[str, int]


def _set_dataset_caption_job(job_id: str, **updates):
    with _dataset_caption_lock:
        job = _dataset_caption_jobs.get(job_id, {})
        job.update(updates)
        _dataset_caption_jobs[job_id] = job


def _run_dataset_caption_job(job_id: str, dataset_id: str):
    try:
        manifest = dataset_module.get_dataset(_datasets_root(), dataset_id)
        pending = [item for item in manifest["images"] if not item.get("caption", "").strip()]
        _set_dataset_caption_job(job_id, status="running", total=len(pending), completed=0)
        if not pending:
            _set_dataset_caption_job(job_id, status="done", message="All images already have captions")
            return

        for index, item in enumerate(pending, start=1):
            with _dataset_caption_lock:
                cancel_event = _dataset_caption_cancel_events.get(job_id)
            if cancel_event and cancel_event.is_set():
                _set_dataset_caption_job(job_id, status="cancelled", message="Captioning cancelled")
                return

            path = dataset_module.image_path(_datasets_root(), dataset_id, item["id"])
            result = caption.auto_caption([str(path)])[0]
            generated_caption = result.get("caption", "").strip()
            if generated_caption:
                dataset_module.update_captions(
                    _datasets_root(),
                    dataset_id,
                    {item["id"]: generated_caption},
                    source=result.get("source", "auto"),
                )
            _set_dataset_caption_job(
                job_id,
                completed=index,
                current_image=item["original_filename"],
            )

        result = dataset_module.get_dataset(_datasets_root(), dataset_id)
        _set_dataset_caption_job(
            job_id,
            status="done",
            completed=len(pending),
            dataset=result,
            message="Captioning complete",
        )
    except Exception as exc:
        _set_dataset_caption_job(job_id, status="error", error=str(exc))


@app.get("/api/datasets")
async def api_list_datasets(user: str = Depends(get_current_user)):
    return {
        "datasets_root": _datasets_root(),
        "datasets": dataset_module.list_datasets(_datasets_root()),
    }


@app.post("/api/datasets")
async def api_create_dataset(
    files: list[UploadFile] = File(...),
    name: str = Form(...),
    dataset_type: str = Form(...),
    trigger_word: str = Form(""),
    sensitive_content: bool = Form(False),
    user: str = Depends(get_current_user),
):
    try:
        uploads = [(file.filename or "image", file.file) for file in files]
        return dataset_module.create_dataset(
            _datasets_root(), name, dataset_type, trigger_word, uploads,
            sensitive_content=sensitive_content,
        )
    except dataset_module.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/import")
async def api_import_dataset(
    file: UploadFile = File(...),
    name: str = Form(""),
    dataset_type: str = Form(""),
    trigger_word: str = Form(""),
    sensitive_content: bool = Form(False),
    user: str = Depends(get_current_user),
):
    try:
        return dataset_module.import_dataset_archive(
            _datasets_root(), file.file, name, dataset_type, trigger_word,
            sensitive_content,
        )
    except dataset_module.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}")
async def api_get_dataset(dataset_id: str, user: str = Depends(get_current_user)):
    try:
        return dataset_module.get_dataset(_datasets_root(), dataset_id)
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/datasets/{dataset_id}")
async def api_update_dataset_settings(
    dataset_id: str,
    payload: DatasetSettingsPayload,
    user: str = Depends(get_current_user),
):
    if _dataset_has_active_job(dataset_id):
        raise HTTPException(
            status_code=409,
            detail="Wait for active dataset jobs to finish before changing settings",
        )
    try:
        return dataset_module.update_dataset_settings(
            _datasets_root(),
            dataset_id,
            payload.name,
            payload.dataset_type,
            payload.trigger_word,
            payload.sensitive_content,
        )
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except dataset_module.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/export")
def api_prepare_dataset_export(
    dataset_id: str,
    user: str = Depends(get_current_user),
):
    file_descriptor, archive_path = tempfile.mkstemp(
        prefix="ai-toolkit-dataset-", suffix=".zip"
    )
    os.close(file_descriptor)
    try:
        export_manifest = dataset_module.create_dataset_export(
            _datasets_root(), dataset_id, archive_path
        )
    except dataset_module.DatasetNotFound as exc:
        _unlink_quietly(archive_path)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except dataset_module.DatasetError as exc:
        _unlink_quietly(archive_path)
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception:
        _unlink_quietly(archive_path)
        raise

    token = secrets.token_urlsafe(32)
    filename = f"{dataset_id}-v{export_manifest['dataset_revision']}.zip"
    with _dataset_export_lock:
        _prune_dataset_exports(time.time())
        _dataset_exports[token] = {
            "path": archive_path,
            "filename": filename,
            "created_at": time.time(),
        }
    expiry_timer = threading.Timer(
        _DATASET_EXPORT_TTL_SECONDS, _expire_dataset_export, args=(token,)
    )
    expiry_timer.daemon = True
    expiry_timer.start()
    return {
        "download_url": f"/api/dataset-exports/{token}",
        "filename": filename,
        "image_count": export_manifest["image_count"],
        "captioned_count": export_manifest["captioned_count"],
    }


@app.get("/api/dataset-exports/{token}")
def api_download_dataset_export(token: str):
    with _dataset_export_lock:
        _prune_dataset_exports(time.time())
        item = _dataset_exports.pop(token, None)
    if item is None:
        raise HTTPException(status_code=404, detail="Dataset export not found or already downloaded")
    try:
        return FileResponse(
            item["path"],
            media_type="application/zip",
            filename=item["filename"],
            background=BackgroundTask(_unlink_quietly, item["path"]),
        )
    except Exception:
        _unlink_quietly(item["path"])
        raise


@app.post("/api/datasets/{dataset_id}/images")
async def api_add_dataset_images(
    dataset_id: str,
    files: list[UploadFile] = File(...),
    user: str = Depends(get_current_user),
):
    try:
        uploads = [(file.filename or "image", file.file) for file in files]
        return dataset_module.add_images(_datasets_root(), dataset_id, uploads)
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except dataset_module.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/datasets/{dataset_id}/images")
async def api_remove_dataset_images(
    dataset_id: str,
    payload: DatasetImageRemovePayload,
    user: str = Depends(get_current_user),
):
    if _dataset_has_active_job(dataset_id):
        raise HTTPException(
            status_code=409,
            detail="Wait for active dataset jobs to finish before removing images",
        )
    try:
        return dataset_module.remove_images(
            _datasets_root(), dataset_id, payload.image_ids
        )
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except dataset_module.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}/images/{image_id}")
async def api_get_dataset_image(
    dataset_id: str,
    image_id: str,
    user: str = Depends(get_current_user),
):
    try:
        return FileResponse(dataset_module.image_path(_datasets_root(), dataset_id, image_id))
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/datasets/{dataset_id}/evaluations/{evaluation_id}/images/{filename}"
)
async def api_get_evaluation_image(
    dataset_id: str,
    evaluation_id: str,
    filename: str,
    user: str = Depends(get_current_user),
):
    try:
        path = dataset_module.evaluation_image_path(
            _datasets_root(), dataset_id, evaluation_id, filename
        )
        return FileResponse(path, media_type="image/png")
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/datasets/{dataset_id}/captions")
async def api_update_dataset_captions(
    dataset_id: str,
    payload: DatasetCaptionPayload,
    user: str = Depends(get_current_user),
):
    try:
        return dataset_module.update_captions(
            _datasets_root(), dataset_id, payload.captions, source="manual"
        )
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except dataset_module.DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/auto-caption")
async def api_auto_caption_dataset(dataset_id: str, user: str = Depends(get_current_user)):
    try:
        manifest = dataset_module.get_dataset(_datasets_root(), dataset_id)
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    with _dataset_caption_lock:
        for job_id, job in _dataset_caption_jobs.items():
            if job.get("dataset_id") == dataset_id and job.get("status") in {"queued", "running"}:
                return {"status": job["status"], "job_id": job_id}
        job_id = uuid.uuid4().hex
        _dataset_caption_jobs[job_id] = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "dataset_name": manifest["name"],
            "status": "queued",
            "completed": 0,
            "total": manifest["analysis"]["image_count"] - manifest["analysis"]["captioned_count"],
        }
        _dataset_caption_cancel_events[job_id] = threading.Event()

    thread = threading.Thread(
        target=_run_dataset_caption_job,
        args=(job_id, dataset_id),
        daemon=True,
    )
    thread.start()
    return {"status": "queued", "job_id": job_id}


@app.get("/api/datasets/caption-progress/{job_id}")
async def api_dataset_caption_progress(job_id: str, user: str = Depends(get_current_user)):
    with _dataset_caption_lock:
        job = _dataset_caption_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Caption job not found")
        return dict(job)


@app.post("/api/datasets/caption-cancel/{job_id}")
async def api_cancel_dataset_caption(job_id: str, user: str = Depends(get_current_user)):
    with _dataset_caption_lock:
        event = _dataset_caption_cancel_events.get(job_id)
        job = _dataset_caption_jobs.get(job_id)
        if not event or not job:
            raise HTTPException(status_code=404, detail="Caption job not found")
        event.set()
    return {"status": "cancelling", "job_id": job_id}


def _set_training_job(job_id: str, **updates):
    with _training_lock:
        job = _training_jobs.get(job_id, {})
        job.update(updates)
        job["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _training_jobs[job_id] = job


def _append_training_log(job_id: str, message: str):
    message = message.strip()
    if not message:
        return
    with _training_lock:
        job = _training_jobs.get(job_id, {})
        logs = job.setdefault("logs", [])
        logs.append(message[-1000:])
        job["logs"] = logs[-100:]
        _training_jobs[job_id] = job


def _iter_process_messages(stream):
    buffer = []
    while True:
        character = stream.read(1)
        if character == "":
            if buffer:
                yield "".join(buffer)
            break
        if character in {"\n", "\r"}:
            if buffer:
                yield "".join(buffer)
                buffer = []
        else:
            buffer.append(character)


def _training_run_record(job: dict) -> dict:
    return {
        key: job.get(key)
        for key in (
            "job_id", "status", "profile", "engine", "created_at", "updated_at",
            "completed_steps", "total_steps", "loss", "output_dir", "lora_path", "error",
            "seed", "recipe", "dataset_revision", "dataset_image_count",
            "dataset_captioned_count",
        )
        if job.get(key) is not None
    }


def _training_was_cancelled(job_id: str) -> bool:
    with _training_lock:
        event = _training_cancel_events.get(job_id)
        return bool(event and event.is_set())


def _active_expansion_job() -> bool:
    with _expansion_lock:
        return any(
            job.get("status") in {"queued", "loading", "running", "cancelling"}
            for job in _expansion_jobs.values()
        )


def _run_krea2_training(job_id: str, dataset_id: str, profile: str, seed: int):
    output_dir = None
    try:
        _set_training_job(job_id, status="preparing", message="Preparing official Krea 2 trainer")
        project_root = os.path.dirname(__file__)
        manifest = dataset_module.get_dataset(_datasets_root(), dataset_id)
        recipe = training.build_krea2_recipe(manifest, profile=profile, seed=seed)
        dataset_snapshot = dataset_module.training_snapshot(manifest)
        trainer_path = training.ensure_krea2_trainer(project_root)
        if _training_was_cancelled(job_id):
            _set_training_job(job_id, status="cancelled", message="Training cancelled")
            return
        dataset_dir = os.path.join(_datasets_root(), dataset_id)
        imagefolder = training.prepare_imagefolder(dataset_dir, manifest, recipe)

        models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
        output_dir = os.path.join(models_path, "lora", "krea2", f"{dataset_id}-{job_id[:8]}")
        os.makedirs(output_dir, exist_ok=False)
        command = training.build_krea2_command(trainer_path, imagefolder, output_dir, recipe)
        with open(os.path.join(output_dir, "training-recipe.json"), "w", encoding="utf-8") as file:
            json.dump({**recipe.to_dict(), **dataset_snapshot}, file, ensure_ascii=False, indent=2)
        if _training_was_cancelled(job_id):
            _set_training_job(job_id, status="cancelled", message="Training cancelled")
            return

        _set_training_job(
            job_id,
            status="running",
            message="Training Krea 2 LoRA on Raw",
            engine=recipe.engine,
            recipe=recipe.to_dict(),
            output_dir=output_dir,
            total_steps=recipe.max_train_steps,
            completed_steps=0,
            **dataset_snapshot,
        )
        with _training_lock:
            started_job = dict(_training_jobs[job_id])
        dataset_module.record_training_run(
            _datasets_root(), dataset_id, _training_run_record(started_job)
        )

        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        popen_kwargs = {
            "cwd": project_root,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 0,
            "env": environment,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **popen_kwargs)
        with _training_lock:
            _training_processes[job_id] = process

        for message in _iter_process_messages(process.stdout):
            _append_training_log(job_id, message)
            progress = training.parse_training_progress(message)
            if progress:
                _set_training_job(job_id, **progress)

        return_code = process.wait()
        with _training_lock:
            current_status = _training_jobs.get(job_id, {}).get("status")
            _training_processes.pop(job_id, None)
        if current_status == "cancelling":
            _set_training_job(job_id, status="cancelled", message="Training cancelled")
        elif return_code != 0:
            raise training.TrainingError(f"Official trainer exited with code {return_code}")
        else:
            lora_path = training.find_final_lora(output_dir)
            if lora_path is None:
                raise training.TrainingError("Training finished but no LoRA weights were produced")
            _set_training_job(
                job_id,
                status="done",
                message="LoRA training complete — ready for Test LoRA",
                completed_steps=recipe.max_train_steps,
                lora_path=str(lora_path),
            )
    except Exception as exc:
        with _training_lock:
            current_status = _training_jobs.get(job_id, {}).get("status")
            _training_processes.pop(job_id, None)
        if current_status == "cancelling":
            _set_training_job(job_id, status="cancelled", message="Training cancelled")
        else:
            _set_training_job(job_id, status="error", error=str(exc), message="Training failed")
    finally:
        with _training_lock:
            job = dict(_training_jobs.get(job_id, {}))
        try:
            dataset_module.record_training_run(
                _datasets_root(), dataset_id, _training_run_record(job)
            )
        except Exception:
            pass


@app.get("/api/datasets/{dataset_id}/training-recipe")
async def api_training_recipe(
    dataset_id: str,
    profile: str = Query("balanced"),
    user: str = Depends(get_current_user),
):
    try:
        manifest = dataset_module.get_dataset(_datasets_root(), dataset_id)
        recipe = training.build_krea2_recipe(manifest, profile=profile)
        return {"recipe": recipe.to_dict(), "dataset_analysis": manifest["analysis"]}
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except training.TrainingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/training")
async def api_start_training(
    dataset_id: str,
    payload: TrainingStartPayload,
    user: str = Depends(get_current_user),
):
    try:
        manifest = dataset_module.get_dataset(_datasets_root(), dataset_id)
        recipe = training.build_krea2_recipe(manifest, profile=payload.profile, seed=payload.seed)
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except training.TrainingError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if not torch.cuda.is_available():
        raise HTTPException(status_code=409, detail="Krea 2 training requires an NVIDIA CUDA GPU")
    try:
        from diffusers import Krea2Pipeline  # noqa: F401
    except ImportError as exc:
        raise HTTPException(
            status_code=409,
            detail="Krea 2 training dependencies are not installed. Run update.sh first.",
        ) from exc
    if _active_expansion_job():
        raise HTTPException(status_code=409, detail="Dataset expansion is currently using the GPU")

    with _training_lock:
        if any(job.get("status") in {"queued", "preparing", "running", "cancelling"} for job in _training_jobs.values()):
            raise HTTPException(status_code=409, detail="Another training job is already active")
        job_id = uuid.uuid4().hex
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _training_jobs[job_id] = {
            "job_id": job_id,
            "dataset_id": dataset_id,
            "dataset_name": manifest["name"],
            "profile": recipe.profile,
            "seed": payload.seed,
            "engine": recipe.engine,
            "status": "queued",
            "message": "Training queued",
            "completed_steps": 0,
            "total_steps": recipe.max_train_steps,
            "created_at": now,
            "updated_at": now,
            "logs": [],
            **dataset_module.training_snapshot(manifest),
        }
        _training_cancel_events[job_id] = threading.Event()

    thread = threading.Thread(
        target=_run_krea2_training,
        args=(job_id, dataset_id, recipe.profile, payload.seed),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued", "recipe": recipe.to_dict()}


@app.get("/api/training/{job_id}")
async def api_training_progress(job_id: str, user: str = Depends(get_current_user)):
    with _training_lock:
        job = _training_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        return dict(job)


@app.post("/api/training/{job_id}/cancel")
async def api_cancel_training(job_id: str, user: str = Depends(get_current_user)):
    with _training_lock:
        job = _training_jobs.get(job_id)
        process = _training_processes.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Training job not found")
        if job.get("status") not in {"queued", "preparing", "running"}:
            return {"status": job.get("status"), "job_id": job_id}
        job["status"] = "cancelling"
        job["message"] = "Cancelling training"
        cancel_event = _training_cancel_events.get(job_id)
        if cancel_event:
            cancel_event.set()
    if process and process.poll() is None:
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
    return {"status": "cancelling", "job_id": job_id}


@app.get("/api/datasets/{dataset_id}/evaluation-preset")
async def api_dataset_evaluation_preset(
    dataset_id: str,
    run_id: str | None = Query(None),
    user: str = Depends(get_current_user),
):
    try:
        manifest = dataset_module.get_dataset(_datasets_root(), dataset_id)
        return evaluation_presets.build_evaluation_preset(manifest, run_id=run_id)
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except evaluation_presets.EvaluationPresetError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/evaluation-verdict")
async def api_dataset_evaluation_verdict(
    dataset_id: str,
    payload: EvaluationVerdictPayload,
    user: str = Depends(get_current_user),
):
    with _evaluation_lock:
        evaluation = _evaluation_progress.get(payload.evaluation_id)
        if evaluation is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        if evaluation.get("status") != "done" or not evaluation.get("result"):
            raise HTTPException(status_code=409, detail="Evaluation is not complete")
        result = evaluation["result"]
        experiment = result.get("experiment")
    try:
        summary = evaluation_presets.summarize_votes(
            experiment, payload.votes, require_complete=True
        )
        comparison = evaluation_presets.comparison_evidence(result)
        normalized_votes = {int(key): int(value) for key, value in payload.votes.items()}
        record = dataset_module.record_evaluation_verdict(
            _datasets_root(), dataset_id, payload.training_run_id,
            payload.evaluation_id, experiment, comparison, normalized_votes, summary,
            OUTPUT_DIR,
        )
        return record
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (dataset_module.DatasetError, evaluation_presets.EvaluationPresetError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _set_expansion_job(job_id: str, **updates):
    with _expansion_lock:
        job = _expansion_jobs.get(job_id, {})
        job.update(updates)
        job["updated_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _expansion_jobs[job_id] = job


def _expansion_run_record(job: dict) -> dict:
    return {
        key: job.get(key)
        for key in (
            "run_id", "job_id", "status", "model", "created_at", "updated_at",
            "candidate_count", "completed", "current_candidate", "message", "error", "candidates",
        )
        if job.get(key) is not None
    }


def _run_dataset_expansion(job_id: str, dataset_id: str, candidate_count: int, seed: int):
    try:
        manifest = dataset_module.get_dataset(_datasets_root(), dataset_id)
        variations = expansion.build_variations(manifest["type"], candidate_count)
        source_images = [
            item for item in manifest.get("images", []) if not item.get("generated")
        ] or manifest.get("images", [])
        source_paths = [
            str(dataset_module.image_path(_datasets_root(), dataset_id, item["id"]))
            for item in source_images
        ]
        source_ids = [item["id"] for item in source_images]
        candidate_dir = dataset_module.expansion_candidate_dir(_datasets_root(), dataset_id, job_id)
        _set_expansion_job(
            job_id,
            status="loading",
            message="Loading Qwen Image Edit Plus — first run may download large model weights",
        )

        for index, variation in enumerate(variations, start=1):
            with _expansion_lock:
                cancel_event = _expansion_cancel_events.get(job_id)
            if cancel_event and cancel_event.is_set():
                raise expansion.ExpansionCancelled("Dataset expansion cancelled")

            if len(source_paths) <= expansion.MAX_REFERENCE_IMAGES:
                refs = source_paths
                ref_ids = source_ids
            else:
                positions = [(index - 1 + offset) % len(source_paths) for offset in range(expansion.MAX_REFERENCE_IMAGES)]
                refs = [source_paths[position] for position in positions]
                ref_ids = [source_ids[position] for position in positions]

            _set_expansion_job(
                job_id,
                status="running",
                message=f"Generating candidate {index} of {candidate_count}",
                current_candidate=index,
                current_step=0,
                total_steps=expansion.INFERENCE_STEPS,
            )

            def progress_cb(step, total, current=index):
                _set_expansion_job(
                    job_id,
                    current_candidate=current,
                    current_step=step,
                    total_steps=total,
                )

            candidate_seed = seed + index - 1
            image = expansion.generate_candidate(
                refs,
                variation.instruction,
                candidate_seed,
                progress_cb=progress_cb,
                cancel_event=cancel_event,
            )
            candidate_id = f"candidate-{index:04d}"
            filename = f"{candidate_id}.png"
            temporary = candidate_dir / f".{candidate_id}.tmp"
            image.save(temporary, format="PNG")
            os.replace(temporary, candidate_dir / filename)
            with _expansion_lock:
                candidates = list(_expansion_jobs[job_id].get("candidates", []))
            candidates.append({
                "id": candidate_id,
                "filename": filename,
                "prompt": variation.instruction,
                "caption": variation.caption,
                "seed": candidate_seed,
                "source_image_ids": ref_ids,
                "status": "pending",
            })
            _set_expansion_job(job_id, completed=index, candidates=candidates)
            with _expansion_lock:
                snapshot = dict(_expansion_jobs[job_id])
            dataset_module.record_expansion_run(
                _datasets_root(), dataset_id, _expansion_run_record(snapshot)
            )

        _set_expansion_job(
            job_id,
            status="done",
            message="Candidates ready — review and accept only the useful images",
            completed=candidate_count,
        )
    except expansion.ExpansionCancelled as exc:
        _set_expansion_job(job_id, status="cancelled", message=str(exc))
    except Exception as exc:
        _set_expansion_job(job_id, status="error", message="Dataset expansion failed", error=str(exc))
    finally:
        expansion.release_pipeline()
        with _expansion_lock:
            snapshot = dict(_expansion_jobs.get(job_id, {}))
        try:
            dataset_module.record_expansion_run(
                _datasets_root(), dataset_id, _expansion_run_record(snapshot)
            )
        except Exception:
            pass


@app.post("/api/datasets/{dataset_id}/expansion")
async def api_start_dataset_expansion(
    dataset_id: str,
    payload: ExpansionStartPayload,
    user: str = Depends(get_current_user),
):
    try:
        manifest = dataset_module.get_dataset(_datasets_root(), dataset_id)
        expansion.build_variations(manifest["type"], payload.candidate_count)
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except expansion.ExpansionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if manifest["analysis"]["source_mode"] != "quick":
        raise HTTPException(status_code=409, detail="Reference expansion is intended for datasets with 1–5 images")
    if payload.seed < 0 or payload.seed > 2_147_483_647 - payload.candidate_count:
        raise HTTPException(status_code=409, detail="Seed is outside the supported range")
    if not torch.cuda.is_available():
        raise HTTPException(status_code=409, detail="Dataset expansion requires an NVIDIA CUDA GPU")
    try:
        from diffusers import QwenImageEditPlusPipeline  # noqa: F401
    except ImportError as exc:
        raise HTTPException(
            status_code=409,
            detail="Qwen Image Edit Plus is unavailable. Run update.sh first.",
        ) from exc
    with _training_lock:
        if any(job.get("status") in {"queued", "preparing", "running", "cancelling"} for job in _training_jobs.values()):
            raise HTTPException(status_code=409, detail="LoRA training is currently using the GPU")
    with _expansion_lock:
        if any(job.get("status") in {"queued", "loading", "running", "cancelling"} for job in _expansion_jobs.values()):
            raise HTTPException(status_code=409, detail="Another dataset expansion job is active")
        job_id = uuid.uuid4().hex
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        _expansion_jobs[job_id] = {
            "run_id": job_id,
            "job_id": job_id,
            "dataset_id": dataset_id,
            "dataset_name": manifest["name"],
            "status": "queued",
            "model": expansion.MODEL_ID,
            "candidate_count": payload.candidate_count,
            "completed": 0,
            "current_candidate": 0,
            "current_step": 0,
            "total_steps": expansion.INFERENCE_STEPS,
            "created_at": now,
            "updated_at": now,
            "message": "Dataset expansion queued",
            "candidates": [],
        }
        _expansion_cancel_events[job_id] = threading.Event()
        snapshot = dict(_expansion_jobs[job_id])
    dataset_module.record_expansion_run(
        _datasets_root(), dataset_id, _expansion_run_record(snapshot)
    )
    threading.Thread(
        target=_run_dataset_expansion,
        args=(job_id, dataset_id, payload.candidate_count, payload.seed),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "queued", "model": expansion.MODEL_ID}


@app.get("/api/expansion/{job_id}")
async def api_expansion_progress(job_id: str, user: str = Depends(get_current_user)):
    with _expansion_lock:
        job = _expansion_jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Expansion job not found")
        return dict(job)


@app.post("/api/expansion/{job_id}/cancel")
async def api_cancel_expansion(job_id: str, user: str = Depends(get_current_user)):
    with _expansion_lock:
        job = _expansion_jobs.get(job_id)
        event = _expansion_cancel_events.get(job_id)
        if not job or not event:
            raise HTTPException(status_code=404, detail="Expansion job not found")
        if job.get("status") not in {"queued", "loading", "running"}:
            return {"job_id": job_id, "status": job.get("status")}
        event.set()
        job["status"] = "cancelling"
        job["message"] = "Cancelling after the current diffusion step"
    return {"job_id": job_id, "status": "cancelling"}


@app.get("/api/datasets/{dataset_id}/expansion/{run_id}/candidates/{candidate_id}")
async def api_expansion_candidate_image(
    dataset_id: str,
    run_id: str,
    candidate_id: str,
    user: str = Depends(get_current_user),
):
    try:
        path = dataset_module.expansion_candidate_path(
            _datasets_root(), dataset_id, run_id, candidate_id
        )
        return FileResponse(path, media_type="image/png")
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/datasets/{dataset_id}/expansion/{run_id}/review")
async def api_review_expansion_candidates(
    dataset_id: str,
    run_id: str,
    payload: ExpansionReviewPayload,
    user: str = Depends(get_current_user),
):
    try:
        return dataset_module.review_expansion_candidates(
            _datasets_root(), dataset_id, run_id, payload.candidate_ids, payload.decision
        )
    except dataset_module.DatasetNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except dataset_module.DatasetError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/image2lora")
async def api_image2lora(
    files: list[UploadFile] = File(...),
    concept: str = Form(...),
    steps: int = Form(50),
    rank: int = Form(16),
    base_model: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    user: str = Depends(get_current_user)
):
    try:
        if len(files) < 1 or len(files) > 5:
            raise HTTPException(status_code=400, detail="Upload 1-5 images")

        image_paths = []
        for f in files:
            data = await f.read()
            path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
            with open(path, "wb") as fp:
                fp.write(data)
            image_paths.append(path)

        output_path = os.path.join(temp_dir, f"{concept}_{uuid.uuid4()}.safetensors")
        lora.image2lora(image_paths, concept, steps, rank, base_model=base_model, output_path=output_path)

        for p in image_paths:
            os.remove(p)

        return FileResponse(output_path, media_type="application/octet-stream", filename=f"{concept}.safetensors")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train_lora")
async def api_train_lora(
    files: list[UploadFile] = File(...),
    concept: str = Form(...),
    steps: int = Form(500),
    rank: int = Form(64),
    captions: str = Form(""),
    base_model: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    user: str = Depends(get_current_user)
):
    try:
        if len(files) < 5:
            raise HTTPException(status_code=400, detail="Upload at least 5 images for full training")

        image_paths = []
        for f in files:
            data = await f.read()
            path = os.path.join(temp_dir, f"{uuid.uuid4()}.png")
            with open(path, "wb") as fp:
                fp.write(data)
            image_paths.append(path)

        captions_list = []
        if captions:
            try:
                captions_list = json.loads(captions)
            except:
                pass

        output_path = os.path.join(temp_dir, f"{concept}_full_{uuid.uuid4()}.safetensors")
        lora.train_lora(image_paths, concept, steps, rank, captions=captions_list, base_model=base_model, output_path=output_path)

        for p in image_paths:
            os.remove(p)

        return FileResponse(output_path, media_type="application/octet-stream", filename=f"{concept}.safetensors")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/merge_lora")
async def api_merge_lora(user: str = Depends(get_current_user),
    files: list[UploadFile] = File(...),
    weights: str = Form(...),
):
    try:
        lora_paths = []
        for f in files:
            data = await f.read()
            path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
            with open(path, "wb") as fp:
                fp.write(data)
            lora_paths.append(path)

        weight_list = [float(w) for w in weights.split(",")]

        if len(lora_paths) != len(weight_list):
            raise HTTPException(status_code=400, detail="Number of files and weights must match")

        output_path = os.path.join(temp_dir, f"merged_{uuid.uuid4()}.safetensors")
        lora.lora_merge(lora_paths, weight_list, output_path)

        for p in lora_paths:
            os.remove(p)

        return FileResponse(output_path, media_type="application/octet-stream", filename="merged.safetensors")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/check_model")
async def api_check_model(
    user: str = Depends(get_current_user),
    model_path: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    vae_path: str = Form(""),
    text_encoder_path: str = Form(""),
):
    missing = []
    warnings = []
    model_type = sdxl._detect_model_type(model_path)

    # Check model file exists
    if model_path.startswith("/") or model_path.startswith("C:"):
        if not os.path.exists(model_path):
            missing.append({"component": "model", "path": model_path, "message": f"Model file not found: {model_path}"})

    if model_type == "zimage":
        zimage_repo = "Tongyi-MAI/Z-Image-Turbo"

        # Check if model is HF repo ID (no local file needed)
        is_hf_repo = not os.path.isfile(model_path) and not os.path.isdir(model_path)

        # Check text encoder (from repo)
        te_found = False
        models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
        local_te_paths = [
            text_encoder_path,
            os.path.join(models_path, "text_encoder"),
        ]
        if os.path.isfile(model_path):
            local_te_paths.insert(1, os.path.join(os.path.dirname(model_path), "text_encoder"))
        for tp in local_te_paths:
            if tp and os.path.exists(tp):
                te_found = True
                break
        if not te_found and not is_hf_repo:
            warnings.append({
                "component": "text_encoder",
                "message": f"Text encoder not found locally. Will download from HuggingFace ({zimage_repo}/text_encoder).",
                "download": f"{zimage_repo}/text_encoder"
            })

        # Check tokenizer (separate tokenizer/ folder, not text_encoder/)
        tok_found = False
        models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
        local_tok_paths = [
            os.path.join(models_path, "tokenizer"),
            os.path.join(models_path, "text_encoder"),
        ]
        if os.path.isfile(model_path):
            local_tok_paths.insert(0, os.path.join(os.path.dirname(model_path), "tokenizer"))
        for tp in local_tok_paths:
            if tp and os.path.exists(tp):
                tok_found = True
                break
        if not tok_found and not is_hf_repo:
            warnings.append({
                "component": "tokenizer",
                "message": f"Tokenizer not found locally. Will download from HuggingFace ({zimage_repo}/tokenizer).",
                "download": f"{zimage_repo}/tokenizer"
            })

        # Check VAE (from repo)
        vae_found = vae_path and os.path.exists(vae_path)
        if not vae_found:
            vae_dirs = [
                os.path.join(models_path, "vae"),
                os.path.join(models_path, "vae_fp16"),
            ]
            if os.path.isfile(model_path):
                vae_dirs.insert(0, os.path.join(os.path.dirname(model_path), "vae"))
            for vp in vae_dirs:
                if os.path.exists(vp):
                    vae_found = True
                    break
        if not vae_found and not is_hf_repo:
            warnings.append({
                "component": "vae",
                "message": f"VAE not found locally. Will download from HuggingFace ({zimage_repo}/vae).",
                "download": f"{zimage_repo}/vae"
            })

    if model_type == "flux2":
        is_hf_repo = not os.path.isfile(model_path) and not os.path.isdir(model_path)

        if not os.path.exists(model_path) and not is_hf_repo:
            missing.append({
                "component": "model",
                "path": model_path,
                "message": f"Model not found.",
            })

    if model_type == "krea2":
        if os.path.isfile(model_path):
            missing.append({
                "component": "model_format",
                "path": model_path,
                "message": (
                    "AI Toolkit needs the full Krea 2 Diffusers repository, not only "
                    "raw.safetensors or turbo.safetensors. Download krea/Krea-2-Raw "
                    "or krea/Krea-2-Turbo as a repository."
                ),
            })
        try:
            from diffusers import Krea2Pipeline  # noqa: F401
        except ImportError:
            missing.append({
                "component": "diffusers",
                "message": "Krea2Pipeline is unavailable. Run update.sh to install the compatible Diffusers build.",
            })

    return {
        "status": "ok" if not missing else "missing",
        "model_type": model_type,
        "missing": missing,
        "warnings": warnings,
    }


def _resolve_lora_paths(lora_paths_json, weight_list):
    """Resolve LoRA paths from dropdown (JSON array of absolute paths)."""
    dropdown_paths = json.loads(lora_paths_json) if lora_paths_json else []
    valid = [p for p in dropdown_paths if os.path.exists(p)]
    if len(valid) < len(weight_list):
        weight_list[:] = weight_list[:len(valid)]
    return valid


def _cleanup_temp_loras(lora_paths, base_dir):
    """Remove only temp LoRA files (those in base_dir)."""
    absp = os.path.abspath(base_dir)
    for p in lora_paths or []:
        if os.path.dirname(os.path.abspath(p)) == absp and os.path.exists(p):
            os.remove(p)


@app.get("/api/lora_files")
async def api_lora_files(user: str = Depends(get_current_user)):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    loras = []
    seen = set()
    if not os.path.exists(models_path):
        return {"lora_files": []}
    for root, dirs, files in os.walk(models_path):
        for fname in sorted(files):
            if not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(root, fname)
            if fpath in seen:
                continue
            seen.add(fpath)
            try:
                keys, _ = _read_safetensors_meta(fpath)
                if keys:
                    role = _detect_role_from_keys(keys)
                    if role == "lora":
                        rel = os.path.relpath(fpath, models_path)
                        loras.append({
                            "name": rel,
                            "path": fpath,
                            "size": os.path.getsize(fpath),
                        })
            except Exception:
                pass
    return {"lora_files": loras}


@app.post("/api/generate")
async def api_generate(
    user: str = Depends(get_current_user),
    prompt: str = Form(...),
    negative: str = Form(""),
    lora_file: List[UploadFile] = File(None),
    lora_paths_json: str = Form("[]"),
    lora_weights: str = Form("[]"),
    model_path: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    vae_path: str = Form(""),
    text_encoder_path: str = Form(""),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    seed: int = Form(42),
    width: int = Form(1024),
    height: int = Form(1024),
):
    try:
        weight_list = json.loads(lora_weights) if lora_weights else []
        lora_paths = _resolve_lora_paths(lora_paths_json, weight_list)
        if not lora_paths and lora_file:
            for f in lora_file:
                data = await f.read()
                path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
                with open(path, "wb") as fh:
                    fh.write(data)
                lora_paths.append(path)

        utils.set_seed(seed)

        img = sdxl_generate(
            prompt=prompt,
            negative=negative,
            lora_paths=lora_paths or None,
            lora_weights=weight_list or None,
            model_path=model_path,
            vae_path=vae_path or None,
            text_encoder_path=text_encoder_path or None,
            steps=steps,
            cfg=cfg,
            seed=seed,
            width=width,
            height=height
        )

        _cleanup_temp_loras(lora_paths, temp_dir)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def sdxl_generate(prompt, negative, lora_paths=None, lora_weights=None, model_path="stabilityai/stable-diffusion-xl-base-1.0", vae_path=None, text_encoder_path=None, steps=20, cfg=7.0, seed=42, width=1024, height=1024):
    return sdxl.sdxl_generate(prompt, negative, lora_paths, lora_weights, model_path, vae_path, text_encoder_path, steps, cfg, seed, width, height)


def _set_gen_progress(gen_id, **kwargs):
    with _gen_lock:
        if gen_id not in _generate_progress:
            _generate_progress[gen_id] = {"status": "loading", "message": "Starting...", "step": 0, "total_steps": 0}
        _generate_progress[gen_id].update(kwargs)


def _get_cancel_event(gen_id):
    with _gen_cancel_lock:
        return _gen_cancel_events.get(gen_id)


def _run_gen(gen_id, prompt, negative, lora_paths, lora_weights, model_path, vae_path, text_encoder_path, steps, cfg, seed, width, height, count=1, mode="queue"):
    cancel_event = threading.Event()
    with _gen_cancel_lock:
        _gen_cancel_events[gen_id] = cancel_event
    gpu_info = gpu.check_gpu()
    dev = gpu_info["gpu_name"] if gpu_info["cuda_available"] else "CPU"
    model_name = os.path.basename(model_path)
    model_size = ""
    if os.path.isfile(model_path):
        size = os.path.getsize(model_path)
        model_size = " (" + _friendly_size(size) + ")"
    start_time = time.time()
    seeds = [seed + i for i in range(count)]
    lora_info = []
    if lora_paths and lora_weights:
        for lp, lw in zip(lora_paths, lora_weights):
            lora_info.append({"path": lp, "weight": lw})
    family = sdxl._detect_model_type(model_path)
    # Resolve default VAE/text_encoder name per family
    if vae_path:
        vae_label = os.path.basename(vae_path)
    else:
        name_lower = os.path.basename(model_path).lower()
        if family == "flux2":
            is_klein = "klein" in name_lower
            vae_label = "black-forest-labs/FLUX.2-klein-9B" if is_klein else "black-forest-labs/FLUX.2-dev-9B"
        elif family == "flux":
            vae_label = "black-forest-labs/FLUX.1-dev"
        elif family == "zimage":
            vae_label = "Tongyi-MAI/Z-Image-Turbo"
        elif family == "krea2":
            vae_label = "Qwen-Image VAE (built-in)"
        elif family == "sdxl":
            vae_label = "stabilityai/sdxl-vae"
        elif family == "sd15":
            vae_label = "runwayml/stable-diffusion-v1-5"
        else:
            vae_label = "built-in"
    if text_encoder_path:
        te_label = os.path.basename(text_encoder_path)
    else:
        if family == "flux2":
            is_klein = "klein" in name_lower
            te_label = "black-forest-labs/FLUX.2-klein-9B" if is_klein else "black-forest-labs/FLUX.2-dev-9B"
        elif family == "zimage":
            te_label = "Tongyi-MAI/Z-Image-Turbo"
        elif family == "krea2":
            te_label = "Qwen3-VL-4B (built-in)"
        else:
            te_label = "built-in"
    _set_gen_progress(gen_id,
        status="loading", message=f"Loading {model_name}{model_size}...",
        images_count=0, total_images=count, dev=dev,
        model_name=model_name, family=family,
        steps=steps, cfg=cfg, seeds=seeds,
        prompt=prompt, negative=negative, width=width, height=height,
        lora=lora_info, start_time=start_time,
        vae=vae_label, text_encoder=te_label)
    try:

        def _on_loading_msg(msg):
            _set_gen_progress(gen_id, status="loading", message=msg)

        def _on_dl_progress(received, total):
            _set_gen_progress(gen_id, dl_received=received, dl_total=total)

        def _save_image(img, idx):
            fname = f"{gen_id}_{idx}.png"
            fpath = os.path.join(OUTPUT_DIR, fname)
            img.save(fpath, format='PNG')
            return f"/output/{fname}"

        if cancel_event.is_set():
            _set_gen_progress(gen_id, status="cancelled", message="Cancelled")
            return

        if mode == "parallel" and count > 1:
            seeds = [seed + i for i in range(count)]
            _set_gen_progress(gen_id, status="generating", step=0, total_steps=steps, message=f"Generating {count} images in parallel...")
            imgs = sdxl.sdxl_generate_parallel(
                prompts=[prompt] * count,
                negative=negative,
                lora_paths=lora_paths, lora_weights=lora_weights,
                model_path=model_path, vae_path=vae_path,
                text_encoder_path=text_encoder_path,
                steps=steps, cfg=cfg, seeds=seeds, width=width, height=height,
                progress_cb=lambda step, total: _set_gen_progress(gen_id, status="generating", step=step, total_steps=total, message=f"Step {step}/{total} ({count} images)"),
                cancel_event=cancel_event,
                on_message=_on_loading_msg,
                on_progress=_on_dl_progress
            )
            image_urls = []
            for i, img in enumerate(imgs):
                image_urls.append(_save_image(img, i))
            with _gen_lock:
                _generate_progress[gen_id]["image_urls"] = image_urls
                _generate_progress[gen_id]["images_count"] = len(image_urls)
            _set_gen_progress(gen_id, status="done", message="Done", elapsed=time.time() - start_time)
        else:
            # Queue: one by one, show each image as it completes
            with _gen_lock:
                _generate_progress[gen_id]["image_urls"] = []
            for i in range(count):
                if cancel_event.is_set():
                    _set_gen_progress(gen_id, status="cancelled", message="Cancelled")
                    return
                _set_gen_progress(gen_id, status="generating", step=0, total_steps=steps, message=f"Image {i+1}/{count}: Loading...", current_image=i+1)
                cur_seed = seed + i
                img = sdxl.sdxl_generate(
                    prompt=prompt, negative=negative,
                    lora_paths=lora_paths, lora_weights=lora_weights,
                    model_path=model_path, vae_path=vae_path,
                    text_encoder_path=text_encoder_path,
                    steps=steps, cfg=cfg, seed=cur_seed, width=width, height=height,
                    progress_cb=lambda step, total, _i=i: _set_gen_progress(gen_id, status="generating", step=step, total_steps=total, message=f"Image {_i+1}/{count}: Step {step}/{total}", current_image=_i+1),
                    cancel_event=cancel_event,
                    on_message=_on_loading_msg,
                    on_progress=_on_dl_progress
                )
                url = _save_image(img, i)
                with _gen_lock:
                    _generate_progress[gen_id]["image_urls"].append(url)
                    _generate_progress[gen_id]["images_count"] = i + 1
            _set_gen_progress(gen_id, status="done", message="Done", elapsed=time.time() - start_time)
    except sdxl.CancelGeneration:
        _set_gen_progress(gen_id, status="cancelled", message="Cancelled")
    except Exception as e:
        _set_gen_progress(gen_id, status="error", message=f"{dev}: {e}")
    finally:
        with _gen_cancel_lock:
            _gen_cancel_events.pop(gen_id, None)


@app.get("/api/generate_progress")
async def api_generate_progress(gen_id: str = Query(...), user: str = Depends(get_current_user)):
    with _gen_lock:
        data = _generate_progress.get(gen_id, {"status": "unknown", "message": "Not found"})
    result = {k: v for k, v in data.items() if k not in ("image_data", "images_data")}
    return result


@app.get("/api/generate_result")
async def api_generate_result(gen_id: str = Query(...), index: int = Query(0), user: str = Depends(get_current_user)):
    with _gen_lock:
        data = _generate_progress.get(gen_id, {})
    image_urls = data.get("image_urls", [])
    if not image_urls or index >= len(image_urls):
        raise HTTPException(status_code=404, detail="Not ready")
    return {"url": image_urls[index]}


@app.post("/api/generate_cancel")
async def api_generate_cancel(gen_id: str = Form(...), user: str = Depends(get_current_user)):
    event = _get_cancel_event(gen_id)
    if event:
        event.set()
        return {"status": "cancelling"}
    return {"status": "not_found"}


ACTIVE_STATUSES = {"loading", "generating", "pending", "downloading"}


@app.get("/api/active_jobs")
async def api_active_jobs(user: str = Depends(get_current_user)):
    jobs = []
    with _gen_lock:
        for gid, data in _generate_progress.items():
            if data.get("status") in ACTIVE_STATUSES:
                jobs.append({
                    "type": "generate",
                    "id": gid,
                    "status": data.get("status"),
                    "message": data.get("message", ""),
                })
    with _evaluation_lock:
        for evaluation_id, data in _evaluation_progress.items():
            if data.get("status") in {"pending", "running", "cancelling"}:
                jobs.append({
                    "type": "evaluation",
                    "id": evaluation_id,
                    "status": data.get("status"),
                    "message": data.get("message", ""),
                })
    with _dataset_caption_lock:
        for job_id, data in _dataset_caption_jobs.items():
            if data.get("status") in {"queued", "running"}:
                jobs.append({
                    "type": "dataset_caption",
                    "id": job_id,
                    "status": data.get("status"),
                    "message": (
                        f"Captioning {data.get('dataset_name', 'dataset')} "
                        f"({data.get('completed', 0)}/{data.get('total', 0)})"
                    ),
                })
    with _training_lock:
        for job_id, data in _training_jobs.items():
            if data.get("status") in {"queued", "preparing", "running", "cancelling"}:
                jobs.append({
                    "type": "training",
                    "id": job_id,
                    "status": data.get("status"),
                    "message": (
                        f"Training {data.get('dataset_name', 'LoRA')} "
                        f"({data.get('completed_steps', 0)}/{data.get('total_steps', 0)})"
                    ),
                })
    with _expansion_lock:
        for job_id, data in _expansion_jobs.items():
            if data.get("status") in {"queued", "loading", "running", "cancelling"}:
                jobs.append({
                    "type": "dataset_expansion",
                    "id": job_id,
                    "status": data.get("status"),
                    "message": (
                        f"Expanding {data.get('dataset_name', 'dataset')} "
                        f"({data.get('completed', 0)}/{data.get('candidate_count', 0)})"
                    ),
                })
    return {"jobs": jobs}


@app.delete("/api/generate_image")
async def api_delete_generate_image(url: str = Query(...), user: str = Depends(get_current_user)):
    """Delete a generated image file."""
    if url.startswith("/output/"):
        fpath = os.path.join(OUTPUT_DIR, os.path.basename(url))
        if os.path.exists(fpath):
            os.remove(fpath)
            return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/api/output_images")
async def api_list_output_images(user: str = Depends(get_current_user)):
    """List all images in the output directory."""
    images = []
    if os.path.exists(OUTPUT_DIR):
        for f in sorted(os.listdir(OUTPUT_DIR), key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x)), reverse=True):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                fpath = os.path.join(OUTPUT_DIR, f)
                images.append({
                    "name": f,
                    "url": f"/output/{f}",
                    "size": os.path.getsize(fpath),
                })
    return {"images": images, "count": len(images)}


@app.delete("/api/output_images")
async def api_delete_all_output_images(user: str = Depends(get_current_user)):
    """Delete all images in the output directory."""
    count = 0
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                os.remove(os.path.join(OUTPUT_DIR, f))
                count += 1
    return {"status": "deleted", "count": count}


@app.post("/api/generate_async")
async def api_generate_async(
    user: str = Depends(get_current_user),
    prompt: str = Form(...),
    negative: str = Form(""),
    lora_file: List[UploadFile] = File(None),
    lora_paths_json: str = Form("[]"),
    lora_weights: str = Form("[]"),
    model_path: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    vae_path: str = Form(""),
    text_encoder_path: str = Form(""),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    seed: int = Form(42),
    width: int = Form(1024),
    height: int = Form(1024),
    count: int = Form(1),
    mode: str = Form("queue"),
):
    gen_id = str(uuid.uuid4())
    count = max(1, min(16, count))
    weight_list = json.loads(lora_weights) if lora_weights else []
    lora_paths = _resolve_lora_paths(lora_paths_json, weight_list)
    if not lora_paths and lora_file:
        for f in lora_file:
            data = await f.read()
            path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
            with open(path, "wb") as fh:
                fh.write(data)
            lora_paths.append(path)
    if len(weight_list) < len(lora_paths):
        weight_list.extend([1.0] * (len(lora_paths) - len(weight_list)))

    utils.set_seed(seed)

    t = threading.Thread(target=_run_gen, args=(gen_id, prompt, negative, lora_paths, weight_list or None, model_path, vae_path or None, text_encoder_path or None, steps, cfg, seed, width, height, count, mode), daemon=True)
    t.start()
    gpu_info = gpu.check_gpu()
    dev = gpu_info["gpu_name"] if gpu_info["cuda_available"] else "CPU"
    model_name = os.path.basename(model_path)
    model_size = ""
    if os.path.isfile(model_path):
        size = os.path.getsize(model_path)
        model_size = " (" + _friendly_size(size) + ")"
    return {"gen_id": gen_id, "status": "started", "device": dev, "model_info": f"Loading {model_name}{model_size}..."}


@app.post("/api/batch_generate")
async def api_batch_generate(
    user: str = Depends(get_current_user),
    prompts: str = Form(...),
    negative: str = Form(""),
    lora_file: List[UploadFile] = File(None),
    lora_paths_json: str = Form("[]"),
    lora_weights: str = Form("[]"),
    model_path: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    vae_path: str = Form(""),
    text_encoder_path: str = Form(""),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    seed: int = Form(42),
    width: int = Form(1024),
    height: int = Form(1024),
):
    try:
        prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]

        weight_list = json.loads(lora_weights) if lora_weights else []
        lora_paths = _resolve_lora_paths(lora_paths_json, weight_list)
        if not lora_paths and lora_file:
            for f in lora_file:
                data = await f.read()
                path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
                with open(path, "wb") as fh:
                    fh.write(data)
                lora_paths.append(path)
        if len(weight_list) < len(lora_paths):
            weight_list.extend([1.0] * (len(lora_paths) - len(weight_list)))

        utils.set_seed(seed)

        images = sdxl.batch_generate(
            prompt_list, negative, lora_paths or None, weight_list or None,
            model_path=model_path,
            vae_path=vae_path or None,
            text_encoder_path=text_encoder_path or None,
            steps=steps, cfg=cfg, seed=seed,
            width=width, height=height
        )

        _cleanup_temp_loras(lora_paths, temp_dir)

        filenames = [f"image_{i+1}.png" for i in range(len(images))]
        zip_data = utils.create_zip_from_images(images, filenames)

        return StreamingResponse(
            io.BytesIO(zip_data),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=images.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/xyz_generate")
async def api_xyz_generate(
    user: str = Depends(get_current_user),
    x_type: str = Form(...),
    x_values: str = Form(...),
    y_type: str = Form(...),
    y_values: str = Form(...),
    z_type: str = Form(""),
    z_values: str = Form(""),
    prompt: str = Form(""),
    negative: str = Form(""),
    lora_file: List[UploadFile] = File(None),
    lora_weights: str = Form("[]"),
    model_path: str = Form(""),
    vae_path: str = Form(""),
    text_encoder_path: str = Form(""),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    width: int = Form(1024),
    height: int = Form(1024),
    count: int = Form(1),
):
    try:
        axes = []
        x_list = [v.strip() for v in x_values.split(",") if v.strip()]
        y_list = [v.strip() for v in y_values.split(",") if v.strip()]
        z_list = [v.strip() for v in z_values.split(",") if v.strip()] if z_type else []

        if not x_list or not y_list:
            raise HTTPException(status_code=400, detail="X and Y axes must have at least one value each.")

        axes.append((x_type, x_list))
        axes.append((y_type, y_list))
        if z_type and z_list:
            axes.append((z_type, z_list))

        lora_paths = []
        weight_list = json.loads(lora_weights) if lora_weights else []
        if lora_file:
            for f in lora_file:
                data = await f.read()
                path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
                with open(path, "wb") as fh:
                    fh.write(data)
                lora_paths.append(path)
        if len(weight_list) < len(lora_paths):
            weight_list.extend([1.0] * (len(lora_paths) - len(weight_list)))

        base_params = {
            'prompt': prompt,
            'negative': negative,
            'model_path': model_path or "stabilityai/stable-diffusion-xl-base-1.0",
            'vae_path': vae_path or None,
            'text_encoder_path': text_encoder_path or None,
            'steps': steps,
            'cfg': cfg,
            'width': width,
            'height': height,
        }

        cells, x_vals, y_vals, z_vals = sdxl.xyz_generate(
            base_params, axes, count=count,
            lora_paths=lora_paths or None,
            lora_weights=weight_list or None,
        )

        for p in lora_paths:
            if os.path.exists(p):
                os.remove(p)

        # Save images and build response
        result_cells = []
        for cell in cells:
            urls = []
            for img_idx, img in enumerate(cell['images']):
                fname = f"xyz_{uuid.uuid4()}.png"
                fpath = os.path.join(OUTPUT_DIR, fname)
                img.save(fpath, format='PNG')
                urls.append(f"/output/{fname}")
            result_cells.append({
                'x': cell['x'],
                'y': cell['y'],
                'z': cell['z'],
                'images': urls,
            })

        return JSONResponse({
            'cells': result_cells,
            'x_labels': x_vals,
            'y_labels': y_vals,
            'z_labels': z_vals if z_type else [],
            'x_type': x_type,
            'y_type': y_type,
            'z_type': z_type,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _resolve_comparison_config(prompts_json, combos_json, lora_paths_json, steps, cfg, count):
    try:
        prompts_list = json.loads(prompts_json)
        combos_list = json.loads(combos_json)
        available_loras = json.loads(lora_paths_json) if lora_paths_json else []
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid evaluation configuration: {exc}")

    prompts_list = [str(prompt).strip() for prompt in prompts_list if str(prompt).strip()]
    if not prompts_list:
        raise HTTPException(status_code=400, detail="At least one prompt required.")
    if not combos_list:
        raise HTTPException(status_code=400, detail="At least one variant required.")
    if len(prompts_list) > 100 or len(combos_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 prompts and 100 variants.")
    if count < 1 or count > 16:
        raise HTTPException(status_code=400, detail="Images per cell must be between 1 and 16.")
    if len(prompts_list) * len(combos_list) * count > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 images per evaluation.")

    resolved_combos = []
    for combo in combos_list:
        model_path = str(combo.get("model", "")).strip()
        if not model_path:
            raise HTTPException(status_code=400, detail="Every variant must select a model.")

        lora_paths = []
        lora_weights = []
        indices = combo.get("loraIndices", [])
        weights = combo.get("loraWeights", [])
        for position, raw_index in enumerate(indices):
            index = int(raw_index)
            if index < 0 or index >= len(available_loras):
                raise HTTPException(status_code=400, detail="A variant references an unknown LoRA.")
            path = str(available_loras[index])
            if not os.path.isfile(path):
                raise HTTPException(status_code=400, detail=f"LoRA file not found: {path}")
            lora_paths.append(path)
            lora_weights.append(float(weights[position]) if position < len(weights) else 1.0)

        combo_steps = int(combo.get("steps") or steps)
        combo_cfg = float(combo.get("cfg") if combo.get("cfg") is not None else cfg)
        if combo_steps < 1 or combo_steps > 100:
            raise HTTPException(status_code=400, detail="Variant steps must be between 1 and 100.")

        resolved_combos.append({
            "label": str(combo.get("label", "")).strip(),
            "model_path": model_path,
            "vae_path": str(combo.get("vae", "")).strip() or None,
            "text_encoder_path": str(combo.get("te", "")).strip() or None,
            "lora_paths": lora_paths or None,
            "lora_weights": lora_weights or None,
            "steps": combo_steps,
            "cfg": combo_cfg,
        })

    return prompts_list, resolved_combos


def _comparison_manifest(prompts, combos, negative, width, height, count, seed):
    return {
        "version": 1,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "base_seed": seed,
        "images_per_cell": count,
        "width": width,
        "height": height,
        "negative_prompt": negative,
        "prompts": prompts,
        "seeds_by_prompt": [
            [seed + (prompt_index * count) + repeat for repeat in range(count)]
            for prompt_index in range(len(prompts))
        ],
        "variants": [
            {
                "label": combo.get("label", ""),
                "model": combo["model_path"],
                "vae": combo.get("vae_path"),
                "text_encoder": combo.get("text_encoder_path"),
                "loras": [
                    {"path": path, "weight": weight}
                    for path, weight in zip(combo.get("lora_paths") or [], combo.get("lora_weights") or [])
                ],
                "steps": combo["steps"],
                "cfg": combo["cfg"],
            }
            for combo in combos
        ],
    }


def _serialize_comparison_cells(cells):
    result_cells = []
    for cell in cells:
        urls = []
        for image in cell["images"]:
            filename = f"eval_{uuid.uuid4()}.png"
            image.save(os.path.join(OUTPUT_DIR, filename), format="PNG")
            urls.append(f"/output/{filename}")
        result_cells.append({
            "x": cell["x"],
            "y": cell["y"],
            "images": urls,
            "seeds": cell.get("seeds", []),
            "steps": cell.get("steps"),
            "cfg": cell.get("cfg"),
        })
    return result_cells


def _set_evaluation_progress(evaluation_id, **updates):
    with _evaluation_lock:
        _evaluation_progress.setdefault(evaluation_id, {}).update(updates)


def _run_comparison_evaluation(
    evaluation_id, prompts, combos, negative, width, height, count, seed, manifest
):
    with _evaluation_cancel_lock:
        cancel_event = _evaluation_cancel_events.setdefault(evaluation_id, threading.Event())
    started_at = time.time()
    total = len(prompts) * len(combos) * count
    _set_evaluation_progress(
        evaluation_id, status="running", message="Loading first variant...",
        completed=0, total=total, started_at=started_at,
    )
    try:
        cells, labels = sdxl.comparison_generate(
            prompts,
            combos,
            negative=negative,
            width=width,
            height=height,
            count=count,
            seed=seed,
            progress_cb=lambda done, amount: _set_evaluation_progress(
                evaluation_id,
                status="running",
                message=f"Generated {done}/{amount} images",
                completed=done,
                total=amount,
            ),
            cancel_event=cancel_event,
        )
        if cancel_event.is_set():
            _set_evaluation_progress(evaluation_id, status="cancelled", message="Evaluation cancelled")
            return

        result = {
            "cells": _serialize_comparison_cells(cells),
            "x_labels": labels,
            "y_labels": prompts,
            "experiment": manifest,
        }
        _set_evaluation_progress(
            evaluation_id,
            status="done",
            message="Evaluation complete",
            completed=total,
            total=total,
            elapsed=time.time() - started_at,
            result=result,
        )
    except sdxl.CancelGeneration:
        _set_evaluation_progress(evaluation_id, status="cancelled", message="Evaluation cancelled")
    except Exception as exc:
        _set_evaluation_progress(evaluation_id, status="error", message=str(exc))
    finally:
        with _evaluation_cancel_lock:
            _evaluation_cancel_events.pop(evaluation_id, None)


@app.post("/api/comparison_generate")
async def api_comparison_generate(
    user: str = Depends(get_current_user),
    prompts: str = Form(...),
    combos: str = Form(...),
    lora_paths_json: str = Form("[]"),
    negative: str = Form(""),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    width: int = Form(1024),
    height: int = Form(1024),
    count: int = Form(1),
    seed: int = Form(42),
):
    prompts_list, resolved_combos = _resolve_comparison_config(
        prompts, combos, lora_paths_json, steps, cfg, count
    )
    cells, combo_labels = sdxl.comparison_generate(
        prompts_list,
        resolved_combos,
        negative=negative,
        width=width,
        height=height,
        count=count,
        seed=seed,
    )
    return JSONResponse({
        "cells": _serialize_comparison_cells(cells),
        "x_labels": combo_labels,
        "y_labels": prompts_list,
        "experiment": _comparison_manifest(
            prompts_list, resolved_combos, negative, width, height, count, seed
        ),
    })


@app.post("/api/comparison_generate_async")
async def api_comparison_generate_async(
    user: str = Depends(get_current_user),
    prompts: str = Form(...),
    combos: str = Form(...),
    lora_paths_json: str = Form("[]"),
    negative: str = Form(""),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    width: int = Form(1024),
    height: int = Form(1024),
    count: int = Form(1),
    seed: int = Form(42),
):
    prompts_list, resolved_combos = _resolve_comparison_config(
        prompts, combos, lora_paths_json, steps, cfg, count
    )
    if width < 64 or height < 64 or width > 4096 or height > 4096:
        raise HTTPException(status_code=400, detail="Width and height must be between 64 and 4096.")

    evaluation_id = str(uuid.uuid4())
    manifest = _comparison_manifest(
        prompts_list, resolved_combos, negative, width, height, count, seed
    )
    _set_evaluation_progress(
        evaluation_id, status="pending", message="Queued", completed=0,
        total=len(prompts_list) * len(resolved_combos) * count,
    )
    with _evaluation_cancel_lock:
        _evaluation_cancel_events[evaluation_id] = threading.Event()
    thread = threading.Thread(
        target=_run_comparison_evaluation,
        args=(
            evaluation_id, prompts_list, resolved_combos, negative,
            width, height, count, seed, manifest,
        ),
        daemon=True,
    )
    thread.start()
    return {"evaluation_id": evaluation_id, "status": "started"}


@app.get("/api/comparison_progress")
async def api_comparison_progress(
    evaluation_id: str = Query(...), user: str = Depends(get_current_user)
):
    with _evaluation_lock:
        data = _evaluation_progress.get(evaluation_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        return {key: value for key, value in data.items() if key != "result"}


@app.get("/api/comparison_result")
async def api_comparison_result(
    evaluation_id: str = Query(...), user: str = Depends(get_current_user)
):
    with _evaluation_lock:
        data = _evaluation_progress.get(evaluation_id)
        if data is None:
            raise HTTPException(status_code=404, detail="Evaluation not found")
        if data.get("status") != "done":
            raise HTTPException(status_code=409, detail="Evaluation is not complete")
        return data["result"]


@app.post("/api/comparison_cancel")
async def api_comparison_cancel(
    evaluation_id: str = Form(...), user: str = Depends(get_current_user)
):
    with _evaluation_cancel_lock:
        event = _evaluation_cancel_events.get(evaluation_id)
    if event is None:
        return {"status": "not_found"}
    event.set()
    _set_evaluation_progress(evaluation_id, status="cancelling", message="Cancelling...")
    return {"status": "cancelling"}


@app.post("/api/lora_info")
async def api_lora_info(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    try:
        data = await file.read()
        path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
        with open(path, "wb") as f:
            f.write(data)

        info = lora.lora_info(path)
        os.remove(path)

        return JSONResponse(info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/extract_lora")
async def api_extract_lora(file: UploadFile = File(...), user: str = Depends(get_current_user)):
    try:
        data = await file.read()
        ext = os.path.splitext(file.filename)[1]
        ckpt_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
        with open(ckpt_path, "wb") as f:
            f.write(data)

        output_path = os.path.join(temp_dir, f"extracted_{uuid.uuid4()}.safetensors")
        lora.extract_lora(ckpt_path, output_path)

        os.remove(ckpt_path)

        return FileResponse(output_path, media_type="application/octet-stream", filename="extracted_lora.safetensors")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upscale")
async def api_upscale(file: UploadFile = File(...), scale: int = Form(4), user: str = Depends(get_current_user)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        upscaled = img_module.upscale(image, scale)

        img_bytes = io.BytesIO()
        upscaled.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/login")
async def check_login(user: str = Depends(get_current_user)):
    return {"logged_in": True, "username": user}


@app.post("/api/user")
async def create_user(
    username: str = Form(...),
    password: str = Form(...),
    admin_user: str = Depends(get_current_user)
):
    conn = sqlite3.connect(DB_FILE)
    admin = conn.execute("SELECT role FROM users WHERE username = ?", (admin_user,)).fetchone()
    if not admin or admin[0] != "admin":
        conn.close()
        raise HTTPException(status_code=403, detail="Admin only")
    hashed = hashlib.sha256(password.encode()).hexdigest()
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        conn.close()
        return {"status": "ok", "message": f"User {username} created"}
    except:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")


@app.post("/api/change_password")
async def change_password(
    new_username: str = Form(...),
    new_password: str = Form(...),
    user: str = Depends(get_current_user)
):
    conn = sqlite3.connect(DB_FILE)
    hashed = hashlib.sha256(new_password.encode()).hexdigest()
    conn.execute("UPDATE users SET username = ?, password = ? WHERE username = ?", (new_username, hashed, user))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": f"Credentials changed to {new_username}"}


@app.get("/api/settings")
async def get_settings(user: str = Depends(get_current_user)):
    return {
        "hf_token": settings.get("hf_token", ""),
        "civitai_token": settings.get("civitai_token", ""),
        "models_path": settings.get("models_path", ""),
        "base_url": settings.get("base_url", "")
    }


@app.post("/api/settings")
async def update_settings(
    hf_token: str = Form(""),
    civitai_token: str = Form(""),
    models_path: str = Form(""),
    base_url: str = Form(""),
    user: str = Depends(get_current_user)
):
    if hf_token:
        settings["hf_token"] = hf_token
    if civitai_token:
        settings["civitai_token"] = civitai_token
    if models_path:
        settings["models_path"] = models_path
    settings["base_url"] = base_url
    save_settings(settings)
    # Update env vars so core/sdxl.py picks up new tokens
    os.environ["HF_TOKEN"] = settings.get("hf_token", "")
    return {"status": "ok", "message": "Settings saved"}


def check_api_key(key: str):
    api_keys = settings.get("api_keys", [])
    for k in api_keys:
        if isinstance(k, dict) and k.get("key") == key:
            return True
        if isinstance(k, str) and k == key:
            return True
    return False


@app.get("/api/settings/api-keys")
async def list_api_keys(user: str = Depends(get_current_user)):
    keys = settings.get("api_keys", [])
    result = []
    for k in keys:
        if isinstance(k, dict):
            result.append({"key": k.get("key", ""), "name": k.get("name", ""), "created": k.get("created", "")})
        else:
            result.append({"key": k, "name": "", "created": ""})
    return {"api_keys": result}


@app.post("/api/settings/api-keys")
async def create_api_key(
    name: str = Form(""),
    user: str = Depends(get_current_user)
):
    api_keys = settings.get("api_keys", [])
    new_key = secrets.token_urlsafe(32)
    now = datetime.datetime.now().isoformat()
    api_keys.append({"key": new_key, "name": name, "created": now})
    settings["api_keys"] = api_keys
    save_settings(settings)
    return {"status": "ok", "key": new_key, "name": name}


@app.delete("/api/settings/api-keys")
async def delete_api_key(
    key: str,
    user: str = Depends(get_current_user)
):
    api_keys = settings.get("api_keys", [])
    new_keys = [k for k in api_keys if (isinstance(k, dict) and k.get("key") != key) or (isinstance(k, str) and k != key)]
    if len(new_keys) == len(api_keys):
        raise HTTPException(status_code=404, detail="Key not found")
    settings["api_keys"] = new_keys
    save_settings(settings)
    return {"status": "ok", "message": "Key deleted"}


def _read_safetensors_meta(path: str):
    try:
        with open(path, 'rb') as f:
            header_len = struct.unpack('<Q', f.read(8))[0]
            if header_len <= 0 or header_len > 50 * 1024 * 1024:
                return None, None
            raw = f.read(header_len)
            if len(raw) != header_len:
                return None, None
            header = json.loads(raw)
        keys = [k for k in header if k != "__metadata__"]
        meta = header.get("__metadata__", {})
        return keys, meta
    except Exception:
        return None, None


def _detect_role_from_keys(tensor_keys):
    if not tensor_keys:
        return "checkpoint"

    keys_lower = [k.lower() for k in tensor_keys]
    joined = ' '.join(keys_lower)

    # LoRA: PEFT (lora_A/lora_B) or Kohya (lora_unet_/lora_te_) format
    if any(k.startswith('lora_unet_') or k.startswith('lora_te_') for k in keys_lower):
        return "lora"
    if any('.lora_a.' in k or '.lora_b.' in k for k in keys_lower):
        return "lora"

    # Checkpoint / UNet (takes priority over VAE/TE since all-in-one exists)
    if 'model.diffusion_model' in joined or 'input_blocks.' in joined:
        return "checkpoint"
    if 'output_blocks.' in joined or 'time_embed' in joined:
        return "checkpoint"

    # VAE
    if all(x in joined for x in ['decoder.conv_in', 'encoder.conv_in']):
        return "vae"
    if ('decoder.mid_block' in joined and 'encoder.mid_block' in joined):
        return "vae"
    if 'quant_conv' in joined or 'post_quant_conv' in joined:
        return "vae"

    # Text Encoder (CLIP)
    if 'text_model.encoder' in joined or 'text_model.final_layer_norm' in joined:
        return "text_encoder"
    if 'token_embedding' in joined or 'positional_embedding' in joined:
        return "text_encoder"

    return "checkpoint"


def _detect_model_family_from_keys(tensor_keys):
    if not tensor_keys:
        return "unknown"

    joined = ' '.join(k.lower() for k in tensor_keys)

    # Z-Image: S3-DiT single stream blocks only (no double stream)
    if 'single_stream_blocks' in joined and 'double_stream' not in joined:
        return "zimage"

    # Z-Image variants: noise_refiner / cap_embedder / context_refiner are unique to Z-Image
    if 'noise_refiner' in joined or 'cap_embedder' in joined or 'context_refiner' in joined:
        return "zimage"

    # SD3: MMDiT joint blocks
    if 'mmdit.' in joined:
        return "sd3"

    # Flux.2: double_blocks + single_blocks (model.diffusion_model. prefix)
    if 'model.diffusion_model.double_blocks' in joined and 'model.diffusion_model.single_blocks' in joined:
        return "flux2"
    # Flux.2 native: double_stream with img_attn or txt_attn
    if 'double_stream' in joined and ('img_attn' in joined or 'txt_attn' in joined):
        return "flux2"

    # Flux.1: transformer_blocks + time_text_embed
    if 'transformer_blocks' in joined and 'time_text_embed' in joined:
        return "flux1"

    # Hunyuan
    if 'hunyuan' in joined:
        return "hunyuan"

    # DiT-based wrapped under model.diffusion_model (e.g. PixArt with diffusers wrapping)
    # Has x_embedder + layers.N instead of input_blocks/mid_block/output_blocks
    if 'model.diffusion_model' in joined:
        # FLUX.2: double_blocks + single_blocks under model.diffusion_model
        if 'model.diffusion_model.double_blocks' in joined and 'model.diffusion_model.single_blocks' in joined:
            return "flux2"
        # Has UNet-specific blocks → SD UNet
        if any(x in joined for x in ['input_blocks.', 'mid_block.', 'output_blocks.']):
            return "sd_unet"
        # Has DiT-specific components (x_embedder + layers) → PixArt variant
        if 'x_embedder' in joined and 'model.diffusion_model.layers.' in joined:
            return "pixart"
        # Fallback: still sd_unet
        return "sd_unet"

    # Unwrapped DiT (original PixArt / DiT format): x_embedder without model prefix
    if 'x_embedder' in joined and 'layers.' in joined:
        return "pixart"

    # PixArt (native format): transformer_blocks + attn1/attn2, no time_text_embed
    # This check comes AFTER wrapped/unwrapped DiT to avoid misclassifying Z-Image as PixArt
    if 'transformer_blocks' in joined and ('attn1' in joined or 'attn2' in joined):
        return "pixart"

    return "unknown"


def _detect_family_from_name(name: str) -> str:
    name_lower = name.lower()
    if any(x in name_lower for x in ["krea-2", "krea_2", "krea2"]):
        return "krea2"
    if any(x in name_lower for x in ["z-image", "z_image", "zimage"]):
        return "zimage"
    if any(x in name_lower for x in ["flux2", "flux.2", "flux-2", "flux_dev2"]):
        return "flux2"
    if any(x in name_lower for x in ["flux", "flux.1", "flux-1"]):
        return "flux1"
    if "sd3" in name_lower or "stable-diffusion-3" in name_lower:
        return "sd3"
    if any(x in name_lower for x in ["hunyuan", "hunyuan"]):
        return "hunyuan"
    if any(x in name_lower for x in ["pixart", "pix-art"]):
        return "pixart"
    if any(x in name_lower for x in ["kolors"]):
        return "kolors"
    if any(x in name_lower for x in ["xl", "sdxl", "pony", "sd_xl", "illustrious"]):
        return "sdxl"
    if any(x in name_lower for x in ["v1-5", "v1.5", "sd15", "sd-1", "runwayml"]):
        return "sd15"
    return ""


def _detect_family(path: str) -> str:
    path = os.path.abspath(path)

    # Diffusers format folder → read model_index.json
    if os.path.isdir(path):
        idx_path = os.path.join(path, "model_index.json")
        if os.path.exists(idx_path):
            try:
                with open(idx_path) as f:
                    idx = json.load(f)
                cls_name = idx.get("_class_name", "")
                mapping = {
                    "StableDiffusionPipeline": "sd15",
                    "StableDiffusionXLPipeline": "sdxl",
                    "StableDiffusion3Pipeline": "sd3",
                    "FluxPipeline": "flux1",
                    "Flux2Pipeline": "flux2",
                    "Flux2KleinPipeline": "flux2",
                    "Flux2KleinKVPipeline": "flux2",
                    "ZImagePipeline": "zimage",
                    "Krea2Pipeline": "krea2",
                    "HunyuanDiTPipeline": "hunyuan",
                    "PixArtAlphaPipeline": "pixart",
                    "KolorsPipeline": "kolors",
                    "LatentConsistencyModelPipeline": "sd15",
                }
                return mapping.get(cls_name, "unknown")
            except Exception:
                pass
        # No model_index.json → check for nested safetensors
        name_fallback = _detect_family_from_name(os.path.basename(path))
        if name_fallback:
            return name_fallback
        for fname in sorted(os.listdir(path)):
            if fname.endswith(".safetensors"):
                keys, _ = _read_safetensors_meta(os.path.join(path, fname))
                if keys:
                    f_result = _detect_model_family_from_keys(keys)
                    if f_result != "unknown":
                        return f_result
                break
        return "unknown"

    # Single file
    if path.endswith(".safetensors"):
        # First check filename for explicit model type
        name_result = _detect_family_from_name(os.path.basename(path))
        if name_result:
            return name_result
        
        # If filename doesn't indicate a specific type, use tensor key detection
        keys, _ = _read_safetensors_meta(path)
        if keys:
            result = _detect_model_family_from_keys(keys)
            if result != "unknown":
                return result

    # Fallback: detect from filename (works for .ckpt, .pt, .pth, and unmatched .safetensors)
    name_result = _detect_family_from_name(os.path.basename(path))
    if name_result:
        return name_result

    return "unknown"


def _detect_model_role(name: str, parent_dir: str = "") -> str:
    name_lower = name.lower()
    parent_lower = parent_dir.lower()

    if "vae" in name_lower or "vae" in parent_lower:
        return "vae"
    if any(x in name_lower for x in ["text_encoder", "text-encoder", "/te", "te_"]):
        return "text_encoder"
    if any(x in parent_lower for x in ["text_encoder", "text-encoder"]):
        return "text_encoder"
    if "lora" in name_lower or "lora" in parent_lower:
        return "lora"

    return "checkpoint"


@app.get("/api/models")
async def list_models(user: str = Depends(get_current_user)):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    result = []
    allowed_ext = {'.ckpt', '.safetensors', '.pt', '.pth'}
    
    if os.path.exists(models_path):
        for item in sorted(os.listdir(models_path)):
            item_path = os.path.join(models_path, item)
            if item.startswith("."):
                continue
            if os.path.isdir(item_path):
                folder_role = _detect_model_role(item)
                folder_files = []
                nested_dirs = []
                for f in sorted(os.listdir(item_path)):
                    f_path = os.path.join(item_path, f)
                    if f.startswith("."):
                        continue
                    if os.path.isdir(f_path):
                        nested_dirs.append(f)
                        # Scan one more level deep for checkpoint files
                        for f2 in sorted(os.listdir(f_path)):
                            f2_path = os.path.join(f_path, f2)
                            if f2.startswith(".") or os.path.isdir(f2_path):
                                continue
                            if os.path.splitext(f2)[1].lower() in allowed_ext:
                                file_role = _detect_model_role(f2, parent_dir=f)
                                if f2.endswith('.safetensors'):
                                    keys, _ = _read_safetensors_meta(f2_path)
                                    if keys:
                                        file_role = _detect_role_from_keys(keys)
                                fe = {"name": f2, "subdir": f, "model_type": file_role}
                                if file_role == "checkpoint":
                                    fe["model_family"] = _detect_family(f2_path)
                                folder_files.append(fe)
                    elif os.path.splitext(f)[1].lower() in allowed_ext:
                        file_role = _detect_model_role(f, parent_dir=item)
                        if f.endswith('.safetensors'):
                            keys, _ = _read_safetensors_meta(f_path)
                            if keys:
                                file_role = _detect_role_from_keys(keys)
                        fe = {"name": f, "model_type": file_role}
                        if file_role == "checkpoint":
                            fe["model_family"] = _detect_family(f_path)
                        folder_files.append(fe)
                entry = {"name": item, "type": "folder", "model_type": folder_role}
                if folder_role == "checkpoint":
                    family = _detect_family(item_path)
                    if family != "unknown":
                        entry["model_family"] = family
                if folder_files:
                    entry["files"] = folder_files[:10]
                if nested_dirs:
                    entry["subdirs"] = nested_dirs
                result.append(entry)
            else:
                ext = os.path.splitext(item)[1].lower()
                if ext in allowed_ext:
                    role = _detect_model_role(item)
                    if item.endswith('.safetensors'):
                        keys, _ = _read_safetensors_meta(item_path)
                        if keys:
                            role = _detect_role_from_keys(keys)
                    fe = {"name": item, "type": "file", "ext": ext, "model_type": role}
                    if role == "checkpoint":
                        fe["model_family"] = _detect_family(item_path)
                    result.append(fe)
    
    return {"models_path": models_path, "models": result}


@app.delete("/api/models")
async def delete_model(
    path: str,
    user: str = Depends(get_current_user)
):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    full_path = os.path.join(models_path, path)
    
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)
        else:
            os.remove(full_path)
        return {"status": "ok", "message": f"Deleted: {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/directories")
async def list_model_directories(user: str = Depends(get_current_user)):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    if not os.path.exists(models_path):
        return {"models_path": models_path, "directories": []}
    dirs = []
    for item in sorted(os.listdir(models_path)):
        item_path = os.path.join(models_path, item)
        if os.path.isdir(item_path) and not item.startswith("."):
            dirs.append(item)
    return {"models_path": models_path, "directories": dirs}


@app.post("/api/models/directories")
async def create_model_directory(
    name: str = Form(...),
    user: str = Depends(get_current_user)
):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    dest = os.path.join(models_path, name)
    if os.path.exists(dest):
        raise HTTPException(status_code=400, detail="Directory already exists")
    os.makedirs(dest, exist_ok=True)
    return {"status": "ok", "message": f"Created directory: {name}"}


@app.get("/api/models/browse")
async def browse_models(path: str = "", user: str = Depends(get_current_user)):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    full = os.path.join(models_path, path) if path else models_path
    if not os.path.exists(full):
        raise HTTPException(status_code=404, detail="Path not found")
    if not os.path.isdir(full):
        raise HTTPException(status_code=400, detail="Not a directory")
    items = []
    for name in sorted(os.listdir(full)):
        if name.startswith("."):
            continue
        item_path = os.path.join(full, name)
        stat = os.stat(item_path)
        rel_path = (path + "/" + name) if path else name
        public_list = settings.get("public_downloads", [])
        items.append({
            "name": name,
            "type": "dir" if os.path.isdir(item_path) else "file",
            "size": stat.st_size,
            "modified": stat.st_mtime,
            "ext": os.path.splitext(name)[1].lower() if os.path.isfile(item_path) else "",
            "public": rel_path in public_list,
        })
    return {
        "models_path": models_path,
        "current_path": path or "",
        "parent_path": "/".join(path.split("/")[:-1]) if path else "",
        "items": items,
    }


@app.post("/api/models/upload")
async def upload_models(
    files: list[UploadFile] = File(...),
    path: str = Form(""),
    user: str = Depends(get_current_user)
):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    dest_dir = os.path.join(models_path, path) if path else models_path
    os.makedirs(dest_dir, exist_ok=True)
    saved = []
    for f in files:
        filepath = os.path.join(dest_dir, f.filename or "unnamed")
        content = await f.read()
        with open(filepath, "wb") as fh:
            fh.write(content)
        saved.append(f.filename or "unnamed")
    return {"status": "ok", "saved": saved}


@app.post("/api/models/rename")
async def rename_model(
    old_path: str = Form(...),
    new_path: str = Form(...),
    user: str = Depends(get_current_user)
):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    src = os.path.join(models_path, old_path)
    dst = os.path.join(models_path, new_path)
    if not os.path.exists(src):
        raise HTTPException(status_code=404, detail="Source not found")
    if os.path.exists(dst):
        raise HTTPException(status_code=400, detail="Destination already exists")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.rename(src, dst)
    return {"status": "ok", "message": f"Renamed to {new_path}"}


@app.post("/api/models/toggle-public")
async def toggle_public(
    path: str = Form(...),
    public: bool = Form(False),
    user: str = Depends(get_current_user)
):
    public_list = settings.get("public_downloads", [])
    if public:
        if path not in public_list:
            public_list.append(path)
    else:
        public_list = [p for p in public_list if p != path]
    settings["public_downloads"] = public_list
    save_settings(settings)
    return {"status": "ok", "public": public}


@app.get("/api/models/download")
async def download_model_file(path: str, api_key: str = ""):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    abs_models = os.path.abspath(models_path)
    abs_file = os.path.abspath(os.path.join(abs_models, path))
    if not abs_file.startswith(abs_models):
        raise HTTPException(status_code=403, detail="Access denied")
    if not os.path.isfile(abs_file):
        raise HTTPException(status_code=404, detail="File not found")

    public = settings.get("public_downloads", [])
    is_public = path in public
    if not is_public and not check_api_key(api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

    return FileResponse(abs_file, filename=os.path.basename(path), media_type="application/octet-stream", headers={"Content-Disposition": f"attachment; filename=\"{os.path.basename(path)}\""})


def unique_path(filepath):
    if not os.path.exists(filepath):
        return filepath
    base, ext = os.path.splitext(filepath)
    i = 1
    while os.path.exists(f"{base}-{i}{ext}"):
        i += 1
    return f"{base}-{i}{ext}"


@app.post("/api/download_model")
async def download_model(
    url: str = Form(...),
    source: str = Form("huggingface"),
    subdirectory: str = Form(""),
    headers_json: str = Form(""),
    save_as: str = Form(""),
    user: str = Depends(get_current_user)
):
    download_id = str(uuid.uuid4())[:8]
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    if subdirectory:
        models_path = os.path.join(models_path, subdirectory)
    os.makedirs(models_path, exist_ok=True)
    
    _download_progress[download_id] = {"total": 0, "received": 0, "status": "pending"}
    
    args = (download_id, url, source, subdirectory, headers_json, save_as, dict(settings), models_path)
    thread = threading.Thread(target=_run_download, args=args, daemon=True)
    thread.start()
    
    return {"status": "started", "download_id": download_id}


def _set_progress(download_id: str, **kw):
    with _dl_lock:
        _download_progress[download_id].update(kw)


def _run_download(download_id: str, url: str, source: str, subdirectory: str, headers_json: str, save_as: str, settings: dict, models_path: str):
    try:
        _set_progress(download_id, status="downloading")
        filepath = _do_download(download_id, url, source, subdirectory, headers_json, save_as, settings, models_path)
        size = os.path.getsize(filepath) if os.path.isfile(filepath) else 0
        _set_progress(download_id, status="done", message=f"Downloaded: {os.path.basename(filepath)} ({_friendly_size(size)})")
    except Exception as e:
        detail = str(e.detail) if hasattr(e, "detail") else str(e)
        _set_progress(download_id, status="error", error=detail)


def _friendly_size(bytes: int) -> str:
    if bytes >= 1073741824:
        return f"{bytes/1073741824:.2f} GB"
    elif bytes >= 1048576:
        return f"{bytes/1048576:.1f} MB"
    elif bytes >= 1024:
        return f"{bytes/1024:.0f} KB"
    return f"{bytes} B"


@app.get("/api/download_progress")
async def download_progress(download_id: str = Query(...)):
    prog = _download_progress.get(download_id)
    if not prog:
        raise HTTPException(status_code=404, detail="Download not found")
    return prog


@app.post("/api/download_hf_subfolder")
async def download_hf_subfolder(
    repo_id: str = Form(...),
    subfolder: str = Form(...),
    target_dir: str = Form(""),
    user: str = Depends(get_current_user),
):
    """Download a subfolder from a HuggingFace repo (e.g. text_encoder, tokenizer, vae)."""
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    if not target_dir:
        target_dir = os.path.join(models_path, subfolder)
    os.makedirs(target_dir, exist_ok=True)

    download_id = str(uuid.uuid4())[:8]
    _download_progress[download_id] = {"total": 0, "received": 0, "status": "pending", "files_done": 0, "files_total": 0}

    args = (download_id, repo_id, subfolder, target_dir, dict(settings))
    thread = threading.Thread(target=_run_download_hf_subfolder, args=args, daemon=True)
    thread.start()

    return {"status": "started", "download_id": download_id, "target_dir": target_dir}


@app.post("/api/download_hf_repo")
async def download_hf_repo(
    repo_id: str = Form(...),
    target_dir: str = Form(""),
    user: str = Depends(get_current_user),
):
    """Download an entire HuggingFace repo to a local directory (e.g. FLUX.2-dev)."""
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    if not target_dir:
        repo_name = repo_id.split("/")[-1]
        target_dir = os.path.join(models_path, repo_name)
    os.makedirs(target_dir, exist_ok=True)

    download_id = str(uuid.uuid4())[:8]
    _download_progress[download_id] = {"total": 0, "received": 0, "status": "pending", "files_done": 0, "files_total": 0}

    args = (download_id, repo_id, target_dir, dict(settings))
    thread = threading.Thread(target=_run_download_hf_repo, args=args, daemon=True)
    thread.start()

    return {"status": "started", "download_id": download_id, "target_dir": target_dir}


def _run_download_hf_repo(download_id: str, repo_id: str, target_dir: str, settings: dict):
    try:
        hf_token = settings.get("hf_token", "") or None
        from huggingface_hub import list_repo_tree, hf_hub_download

        _set_progress(download_id, status="listing")

        # List all files in the repo
        all_items = list(list_repo_tree(repo_id, recursive=True))
        files = [item for item in all_items if hasattr(item, 'size') and item.size is not None]

        _set_progress(download_id, files_total=len(files), files_done=0, total=0, received=0, status="downloading")

        for i, file_obj in enumerate(files):
            rel_path = file_obj.path
            filename = os.path.basename(rel_path)
            # Preserve subdirectory structure
            rel_dir = os.path.dirname(rel_path)
            dest_dir = os.path.join(target_dir, rel_dir) if rel_dir else target_dir
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, filename)

            _set_progress(download_id, current_file=rel_path, files_done=i)

            # Skip if already exists and correct size
            if os.path.isfile(dest) and os.path.getsize(dest) == file_obj.size:
                continue

            try:
                cached = hf_hub_download(
                    repo_id=repo_id,
                    filename=rel_path,
                    token=hf_token,
                    local_dir=target_dir,
                )
                # Move to target if different location
                if os.path.abspath(cached) != os.path.abspath(dest):
                    import shutil
                    shutil.move(cached, dest)
            except Exception as e:
                print(f"[download_hf_repo] Warning: failed to download {rel_path}: {e}")

        _set_progress(download_id, files_done=len(files), status="done", message=f"Downloaded {repo_id}")

    except Exception as e:
        detail = str(e.detail) if hasattr(e, "detail") else str(e)
        _set_progress(download_id, status="error", error=detail)


def _run_download_hf_subfolder(download_id: str, repo_id: str, subfolder: str, target_dir: str, settings: dict):
    try:
        hf_token = settings.get("hf_token", "") or None
        from huggingface_hub import list_repo_tree, hf_hub_download

        _set_progress(download_id, status="listing")

        # List all files in subfolder
        all_items = list(list_repo_tree(repo_id, path_in_repo=subfolder, recursive=True))
        files = [item for item in all_items if hasattr(item, 'size') and item.size is not None]

        _set_progress(download_id, files_total=len(files), files_done=0, total=0, received=0, status="downloading")

        for i, file_obj in enumerate(files):
            rel_path = file_obj.path
            filename = os.path.basename(rel_path)
            dest = os.path.join(target_dir, filename)

            _set_progress(download_id, current_file=filename, files_done=i)

            # Skip if already exists and correct size
            if os.path.isfile(dest) and os.path.getsize(dest) == file_obj.size:
                continue

            try:
                cached = hf_hub_download(
                    repo_id=repo_id,
                    filename=rel_path,
                    token=hf_token,
                    local_dir=os.path.dirname(target_dir),
                )
                # Move to target if different location
                if os.path.abspath(cached) != os.path.abspath(dest):
                    import shutil
                    shutil.move(cached, dest)
            except Exception as e:
                print(f"[download_hf_subfolder] Warning: failed to download {rel_path}: {e}")

        _set_progress(download_id, files_done=len(files), status="done", message=f"Downloaded {subfolder}")

    except Exception as e:
        detail = str(e.detail) if hasattr(e, "detail") else str(e)
        _set_progress(download_id, status="error", error=detail)


def _do_download(download_id: str, url: str, source: str, subdirectory: str, headers_json: str, save_as: str, settings: dict, models_path: str) -> str:
    if source == "huggingface":
        hf_token = settings.get("hf_token", "")
        
        # Parse HuggingFace URL
        hf_url = url.replace("https://huggingface.co/", "").replace("https://HF.co/", "").strip("/")
        for sep in ("/blob/", "/resolve/", "/tree/"):
            if sep in hf_url:
                repo_id, after = hf_url.split(sep, 1)
                parts = after.split("/", 1)
                filename = parts[1] if len(parts) > 1 else ""
                break
        else:
            repo_id = hf_url
            filename = ""
            repo_parts = repo_id.split("/")
            if len(repo_parts) > 2:
                repo_id = "/".join(repo_parts[:2])
                filename = "/".join(repo_parts[2:])
        
        base = models_path
        
        if filename:
            just_filename = filename.split("/")[-1] if "/" in filename else filename
            if save_as:
                just_filename = save_as
            dest = base
            os.makedirs(dest, exist_ok=True)
            filepath = unique_path(os.path.join(dest, just_filename))
            
            if shutil.which("hf"):
                cmd = ["hf", "download", repo_id, filename, "--local-dir", dest]
                if hf_token:
                    cmd.extend(["--token", hf_token])
            else:
                cmd = ["huggingface-cli", "download", repo_id, filename, "--local-dir", dest]
                if hf_token:
                    cmd.extend(["--token", hf_token])
            
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=504, detail="Download timed out after 1 hour")
            
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                detail = r.stderr.strip() or r.stdout.strip() or "File not found after download"
                raise HTTPException(status_code=500, detail=detail)
            return filepath
        else:
            dest_name = save_as if save_as else repo_id.replace("/", "_")
            dest = os.path.join(base, dest_name)
            
            if shutil.which("hf"):
                cmd = ["hf", "download", repo_id, "--local-dir", dest]
                if hf_token:
                    cmd.extend(["--token", hf_token])
            else:
                cmd = ["huggingface-cli", "download", repo_id, "--local-dir", dest]
                if hf_token:
                    cmd.extend(["--token", hf_token])
            
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=504, detail="Download timed out after 1 hour")
            
            if not os.path.isdir(dest) or not any(True for _ in os.scandir(dest)):
                detail = r.stderr.strip() or r.stdout.strip() or "Download directory is empty"
                raise HTTPException(status_code=500, detail=detail)
            return dest
    
    elif source == "civitai":
        civit_token = settings.get("civitai_token", "")
        
        import requests
        import urllib.parse
        
        parsed = urllib.parse.urlparse(url)
        domain = parsed.hostname or "civitai.com"
        download_host = f"https://{domain}"
        search_api = "https://civitai.com/api/v1/models"
        dl_headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        
        qs = urllib.parse.parse_qs(parsed.query)
        version_id = qs.get("modelVersionId", [None])[0]
        model_name = None
        model_id_from_path = None
        
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) >= 2 and path_parts[0] == "models":
            raw_name = path_parts[-1].replace("-", " ").replace("_", " ").title()
            if raw_name:
                model_name = raw_name
            if not version_id and len(path_parts) >= 2:
                try:
                    model_id_from_path = int(path_parts[1])
                except ValueError:
                    model_id_from_path = None
        
        if version_id:
            download_url = f"{download_host}/api/download/models/{version_id}"
            if civit_token:
                download_url += f"?token={civit_token}"
            model_name = model_name or f"model-{version_id}"
        elif model_id_from_path:
            download_url = f"{download_host}/api/download/models/{model_id_from_path}"
            if civit_token:
                download_url += f"?token={civit_token}"
            model_name = model_name or f"model-{model_id_from_path}"
        else:
            resp = requests.get(f"{search_api}?search={url}", headers=dl_headers, timeout=30)
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail="CivitAI API error")
            data = resp.json()
            if not data.get("items"):
                raise HTTPException(status_code=404, detail="Model not found")
            model_id = data["items"][0]["id"]
            version_id = data["items"][0]["modelVersions"][0]["id"]
            model_name = data["items"][0]["name"]
            download_url = f"{download_host}/api/download/models/{version_id}"
            if civit_token:
                download_url += f"?token={civit_token}"
        
        base_name = save_as if save_as else f"{model_name.replace(' ', '_')}.safetensors"
        filepath = unique_path(os.path.join(models_path, base_name))
        
        try:
            _set_progress(download_id, total=0)
            head = requests.head(download_url, headers=dl_headers, timeout=30)
            total = int(head.headers.get("Content-Length", 0))
            if total:
                _set_progress(download_id, total=total)
            
            r = requests.get(download_url, headers=dl_headers, timeout=7200, stream=True)
            ctype = r.headers.get("Content-Type", "")
            if r.status_code == 404:
                raise HTTPException(status_code=500, detail="Model not found on CivitAI")
            if r.status_code != 200 or "text/html" in ctype:
                raise HTTPException(status_code=500, detail=f"Download failed: server returned {r.status_code} (HTML page) — the model may require you to be logged in to CivitAI")
            if not total:
                total = int(r.headers.get("Content-Length", 0))
            _set_progress(download_id, total=total)
            received = 0
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        received += len(chunk)
                        if total:
                            _set_progress(download_id, received=received)
            if total and received < total:
                raise HTTPException(status_code=500, detail=f"Download incomplete: {received}/{total} bytes")
            _set_progress(download_id, received=received)
            return filepath
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    else:  # other - direct URL
        try:
            import requests
            extra_headers = {}
            if headers_json:
                extra_headers = json.loads(headers_json)
            model_name = save_as if save_as else (url.split("/")[-1] or "model")
            filepath = unique_path(os.path.join(models_path, model_name))
            os.makedirs(models_path, exist_ok=True)
            
            _set_progress(download_id, total=0)
            try:
                head = requests.head(url, headers=extra_headers, timeout=30)
                total = int(head.headers.get("Content-Length", 0))
                if total:
                    _set_progress(download_id, total=total)
            except Exception:
                total = 0
            
            wget_available = shutil.which("wget") is not None
            curl_available = shutil.which("curl") is not None
            
            if wget_available:
                cmd = ["wget", "-O", filepath, url]
                for k, v in extra_headers.items():
                    cmd.extend(["--header", f"{k}: {v}"])
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                if r.returncode != 0:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    detail = r.stderr.strip() or r.stdout.strip() or f"wget exited with code {r.returncode}"
                    raise HTTPException(status_code=500, detail=detail)
                if os.path.exists(filepath):
                    _set_progress(download_id, received=os.path.getsize(filepath))
            elif curl_available:
                cmd = ["curl", "-L", "-o", filepath, url]
                for k, v in extra_headers.items():
                    cmd.extend(["-H", f"{k}: {v}"])
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
                if r.returncode != 0:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    detail = r.stderr.strip() or r.stdout.strip() or f"curl exited with code {r.returncode}"
                    raise HTTPException(status_code=500, detail=detail)
                if os.path.exists(filepath):
                    _set_progress(download_id, received=os.path.getsize(filepath))
            else:
                r = requests.get(url, headers=extra_headers, stream=True, timeout=(30, 3600))
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                if total:
                    _set_progress(download_id, total=total)
                tmp = filepath + ".partial"
                received = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            received += len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                if total and received < total:
                    os.remove(tmp)
                    raise HTTPException(status_code=500, detail=f"Download incomplete: {received}/{total} bytes")
                os.replace(tmp, filepath)
                _set_progress(download_id, received=received)
            
            if not os.path.exists(filepath):
                raise HTTPException(status_code=500, detail="File not found after download")
            size = os.path.getsize(filepath)
            if size == 0:
                os.remove(filepath)
                raise HTTPException(status_code=500, detail="Downloaded file is empty (0 bytes)")
            return filepath
        except HTTPException:
            raise
        except subprocess.TimeoutExpired:
            if os.path.exists(filepath):
                os.remove(filepath)
            raise HTTPException(status_code=504, detail="Download timed out after 2 hours")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=500, detail="Unknown error")


@app.get("/api/update")
async def update_env(user: str = Depends(get_current_user)):
    runtime = "bare"
    if os.path.exists("/.dockerenv") or os.environ.get("container", ""):
        runtime = "docker"
    elif os.environ.get("PM2_HOME") or os.environ.get("PM2_PROCESS_WATCH"):
        runtime = "pm2"
    app_dir = os.path.dirname(os.path.abspath(__file__))
    git_hash = ""
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=app_dir, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass
    return {
        "runtime": runtime,
        "git_hash": git_hash,
        "app_dir": app_dir,
        "pm2_restart": runtime == "pm2",  # PM2 watches file changes, auto-restarts
    }


@app.post("/api/update")
async def update_app(user: str = Depends(get_current_user)):
    import subprocess
    import shutil

    app_dir = os.path.dirname(os.path.abspath(__file__))

    # Detect runtime
    runtime = "bare"
    if os.path.exists("/.dockerenv") or os.environ.get("container", ""):
        runtime = "docker"
    elif os.environ.get("PM2_HOME") or os.environ.get("PM2_PROCESS_WATCH"):
        runtime = "pm2"

    if not shutil.which("git"):
        raise HTTPException(status_code=400, detail="git not found on this server")

    try:
        result = subprocess.run(
            ["git", "pull"],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="git pull timed out")

    output = result.stdout + result.stderr

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=output or "git pull failed")

    already_updated = "Already up to date" in output

    restart_msg = ""
    if not already_updated:
        if runtime == "docker":
            restart_msg = "Running inside Docker. Please restart the container manually."
        elif runtime == "pm2":
            try:
                pm2_name = os.environ.get("name", "")
                if not pm2_name:
                    pm2_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
                subprocess.run(["pm2", "restart", pm2_name], capture_output=True, text=True, timeout=15, cwd=app_dir)
                restart_msg = f"PM2 process '{pm2_name}' restarted."
            except Exception:
                restart_msg = "PM2 detected but restart failed. Please restart PM2 manually."
        else:
            restart_msg = "Bare process — please restart the server manually."

    return {
        "output": output,
        "runtime": runtime,
        "restart": restart_msg,
        "already_updated": already_updated,
    }


if __name__ == "__main__":
    import uvicorn
    import logging

    class _SuppressPollFilter(logging.Filter):
        def filter(self, record):
            msg = record.getMessage()
            if '"/api/generate_progress' in msg or '"/api/active_jobs' in msg:
                return False
            return True

    for _name in ("uvicorn.access", "uvicorn"):
        logging.getLogger(_name).addFilter(_SuppressPollFilter())

    # Ensure component folders exist in models directory
    _models_dir = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    for _sub in ("text_encoder", "vae", "tokenizer"):
        os.makedirs(os.path.join(_models_dir, _sub), exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=7800, timeout_keep_alive=600, timeout_graceful_shutdown=600)
