import os
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
from typing import List

import torch
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
from contextlib import asynccontextmanager

from core import gpu, caption, sdxl, lora, image as img_module, utils
import threading
import time

security = HTTPBasic()

_download_progress: dict = {}
_dl_lock = threading.Lock()

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    gpu_info = gpu.check_gpu()
    print(f"👑 ai-toolkit ready on GB10 | GPU: {gpu_info['gpu_name']} | VRAM: {gpu_info['vram_total_gb']}GB")
    yield


app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")


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


@app.post("/api/generate")
async def api_generate(
    user: str = Depends(get_current_user),
    prompt: str = Form(...),
    negative: str = Form(""),
    lora_file: List[UploadFile] = File(None),
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
        lora_paths = []
        weight_list = json.loads(lora_weights) if lora_weights else []
        if lora_file:
            for i, f in enumerate(lora_file):
                data = await f.read()
                path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
                with open(path, "wb") as fh:
                    fh.write(data)
                lora_paths.append(path)
        if len(weight_list) < len(lora_paths):
            weight_list.extend([1.0] * (len(lora_paths) - len(weight_list)))

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

        for p in lora_paths:
            if os.path.exists(p):
                os.remove(p)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def sdxl_generate(prompt, negative, lora_paths=None, lora_weights=None, model_path="stabilityai/stable-diffusion-xl-base-1.0", vae_path=None, text_encoder_path=None, steps=20, cfg=7.0, seed=42, width=1024, height=1024):
    return sdxl.sdxl_generate(prompt, negative, lora_paths, lora_weights, model_path, vae_path, text_encoder_path, steps, cfg, seed, width, height)


@app.post("/api/batch_generate")
async def api_batch_generate(
    user: str = Depends(get_current_user),
    prompts: str = Form(...),
    negative: str = Form(""),
    lora_file: List[UploadFile] = File(None),
    lora_weights: str = Form("[]"),
    model_path: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    vae_path: str = Form(""),
    text_encoder_path: str = Form(""),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    seed: int = Form(42),
):
    try:
        prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]

        lora_paths = []
        weight_list = json.loads(lora_weights) if lora_weights else []
        if lora_file:
            for i, f in enumerate(lora_file):
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
            steps=steps, cfg=cfg, seed=seed
        )

        for p in lora_paths:
            if os.path.exists(p):
                os.remove(p)

        filenames = [f"image_{i+1}.png" for i in range(len(images))]
        zip_data = utils.create_zip_from_images(images, filenames)

        return StreamingResponse(
            io.BytesIO(zip_data),
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=images.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

    # SD3: MMDiT joint blocks
    if 'mmdit.' in joined:
        return "sd3"

    # SD UNet family (SD1.5 / SDXL)
    if 'model.diffusion_model' in joined:
        return "sd_unet"

    # Flux.2: double_stream with img_attn or txt_attn
    if 'double_stream' in joined and ('img_attn' in joined or 'txt_attn' in joined):
        return "flux2"

    # Flux.1: transformer_blocks + time_text_embed
    if 'transformer_blocks' in joined and 'time_text_embed' in joined:
        return "flux1"

    # Hunyuan
    if 'hunyuan' in joined:
        return "hunyuan"

    # PixArt (no time_text_embed, just transformer_blocks)
    if 'transformer_blocks' in joined and ('attn1' in joined or 'attn2' in joined):
        return "pixart"

    return "unknown"


def _detect_family_from_name(name: str) -> str:
    name_lower = name.lower()
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
                    "ZImagePipeline": "zimage",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7800, timeout_keep_alive=600, timeout_graceful_shutdown=600)