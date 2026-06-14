import os
import tempfile
import uuid
import json
import io
import sqlite3
import hashlib
import subprocess

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from PIL import Image
from contextlib import asynccontextmanager

from core import gpu, caption, sdxl, lora, image as img_module, utils

security = HTTPBasic()

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
        "models_path": os.path.join(os.path.expanduser("~"), "models")
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
    lora_file: UploadFile = File(None),
    model_path: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    vae_path: str = Form(""),
    text_encoder_path: str = Form(""),
    steps: int = Form(20),
    seed: int = Form(42),
    width: int = Form(1024),
    height: int = Form(1024),
):
    try:
        lora_path = None
        if lora_file:
            lora_data = await lora_file.read()
            lora_path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
            with open(lora_path, "wb") as f:
                f.write(lora_data)

        utils.set_seed(seed)

        img = sdxl_generate(
            prompt=prompt,
            negative=negative,
            lora_path=lora_path,
            model_path=model_path,
            vae_path=vae_path or None,
            text_encoder_path=text_encoder_path or None,
            steps=steps,
            seed=seed,
            width=width,
            height=height
        )

        if lora_path and os.path.exists(lora_path):
            os.remove(lora_path)

        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return StreamingResponse(img_bytes, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def sdxl_generate(prompt, negative, lora_path, model_path, vae_path, text_encoder_path, steps, seed, width, height):
    return sdxl.sdxl_generate(prompt, negative, lora_path, model_path, vae_path, text_encoder_path, steps, seed, width, height)


@app.post("/api/batch_generate")
async def api_batch_generate(
    user: str = Depends(get_current_user),
    prompts: str = Form(...),
    negative: str = Form(""),
    lora_file: UploadFile = File(None),
    model_path: str = Form("stabilityai/stable-diffusion-xl-base-1.0"),
    vae_path: str = Form(""),
    text_encoder_path: str = Form(""),
    steps: int = Form(20),
    seed: int = Form(42),
):
    try:
        prompt_list = [p.strip() for p in prompts.split("\n") if p.strip()]

        lora_path = None
        if lora_file:
            lora_data = await lora_file.read()
            lora_path = os.path.join(temp_dir, f"{uuid.uuid4()}.safetensors")
            with open(lora_path, "wb") as f:
                f.write(lora_data)

        utils.set_seed(seed)

        images = sdxl.batch_generate(
            prompt_list, negative, lora_path,
            model_path=model_path,
            vae_path=vae_path or None,
            text_encoder_path=text_encoder_path or None,
            steps=steps, seed=seed
        )

        if lora_path and os.path.exists(lora_path):
            os.remove(lora_path)

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
        "models_path": settings.get("models_path", "")
    }


@app.post("/api/settings")
async def update_settings(
    hf_token: str = Form(""),
    civitai_token: str = Form(""),
    models_path: str = Form(""),
    user: str = Depends(get_current_user)
):
    if hf_token:
        settings["hf_token"] = hf_token
    if civitai_token:
        settings["civitai_token"] = civitai_token
    if models_path:
        settings["models_path"] = models_path
    save_settings(settings)
    return {"status": "ok", "message": "Settings saved"}


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
                folder_files = []
                nested_dirs = []
                for f in sorted(os.listdir(item_path)):
                    f_path = os.path.join(item_path, f)
                    if f.startswith("."):
                        continue
                    if os.path.isdir(f_path):
                        nested_dirs.append(f)
                    elif os.path.splitext(f)[1].lower() in allowed_ext:
                        folder_files.append(f)
                entry = {"name": item, "type": "folder"}
                if folder_files:
                    entry["files"] = folder_files[:10]
                if nested_dirs:
                    entry["subdirs"] = nested_dirs
                result.append(entry)
            else:
                ext = os.path.splitext(item)[1].lower()
                if ext in allowed_ext:
                    result.append({"name": item, "type": "file", "ext": ext})
    
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
            import shutil
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


@app.post("/api/download_model")
async def download_model(
    url: str = Form(...),
    source: str = Form("huggingface"),
    subdirectory: str = Form(""),
    headers_json: str = Form(""),
    user: str = Depends(get_current_user)
):
    models_path = settings.get("models_path", os.path.join(os.path.expanduser("~"), "models"))
    if subdirectory:
        models_path = os.path.join(models_path, subdirectory)
    os.makedirs(models_path, exist_ok=True)
    
    result = {"status": "ok", "message": ""}
    
    if source == "huggingface":
        hf_token = settings.get("hf_token", "")
        model_name = url.replace("https://huggingface.co/", "").replace("https://HF.co/", "").strip("/")
        dest = os.path.join(models_path, model_name.replace("/", "_"))
        
        cmd = ["huggingface-cli", "download", model_name, "--local-dir", dest]
        if hf_token:
            cmd.extend(["--token", hf_token])
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            result["message"] = f"Downloaded to {dest}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")
    
    elif source == "civitai":
        civit_token = settings.get("civitai_token", "")
        api_url = "https://civitai.com/api/v1/models"
        
        import requests
        headers = {"Authorization": f"Bearer {civit_token}"} if civit_token else {}
        
        try:
            response = requests.get(f"{api_url}?search={url}", headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data["items"]:
                    model_id = data["items"][0]["id"]
                    version_id = data["items"][0]["modelVersions"][0]["id"]
                    
                    download_url = f"{api_url}/{model_id}/download?token={civit_token}" if civit_token else f"{api_url}/{model_id}/download"
                    
                    model_name = data["items"][0]["name"]
                    dest = os.path.join(models_path, f"{model_name.replace(' ', '_')}")
                    os.makedirs(dest, exist_ok=True)
                    
                    r = requests.get(download_url, headers=headers)
                    if r.status_code == 200:
                        filename = os.path.join(dest, f"{model_name}.safetensors")
                        with open(filename, "wb") as f:
                            f.write(r.content)
                        result["message"] = f"Downloaded to {filename}"
                    else:
                        raise HTTPException(status_code=500, detail=f"Download failed: {r.status_code}")
                else:
                    raise HTTPException(status_code=404, detail="Model not found")
            else:
                raise HTTPException(status_code=500, detail="CivitAI API error")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    else:  # other - direct URL
        try:
            import shutil
            extra_headers = {}
            if headers_json:
                import json
                extra_headers = json.loads(headers_json)
            model_name = url.split("/")[-1] or "model"
            filepath = os.path.join(models_path, model_name)
            os.makedirs(models_path, exist_ok=True)
            
            wget_available = shutil.which("wget") is not None
            curl_available = shutil.which("curl") is not None
            
            if wget_available:
                cmd = ["wget", "-O", filepath, url]
                for k, v in extra_headers.items():
                    cmd.extend(["--header", f"{k}: {v}"])
                subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            elif curl_available:
                cmd = ["curl", "-L", "-o", filepath, url]
                for k, v in extra_headers.items():
                    cmd.extend(["-H", f"{k}: {v}"])
                subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
            else:
                import requests
                r = requests.get(url, headers=extra_headers, stream=True, timeout=(30, 3600))
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                tmp = filepath + ".partial"
                with open(tmp, "wb") as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                if total and downloaded < total:
                    os.remove(tmp)
                    raise HTTPException(status_code=500, detail=f"Download incomplete: {downloaded}/{total} bytes")
                os.replace(tmp, filepath)
            
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                raise HTTPException(status_code=500, detail="File missing or empty after save")
            result["message"] = f"Downloaded to {filepath}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7800, timeout_keep_alive=600, timeout_graceful_shutdown=600)