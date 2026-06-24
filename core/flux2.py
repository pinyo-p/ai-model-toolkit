import torch
import os
import time
import traceback
from safetensors import safe_open
import huggingface_hub as hf_hub
from diffusers import (
    Flux2KleinPipeline,
    Flux2KleinKVPipeline,
    Flux2Pipeline,
    AutoencoderKLFlux2,
    Flux2Transformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast


def load_base_flux2_and_swap_weights(model_path, dtype, hf_token, on_message=None, on_progress=None, cancel_event=None):
    """Load base Flux2 pipeline from HF, then swap transformer weights from a single checkpoint file."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    # Pick pipeline class from filename
    name_lower = os.path.basename(model_path).lower()
    is_klein = "klein" in name_lower
    pipeline_cls = Flux2KleinPipeline if is_klein else Flux2Pipeline
    pipe_name = "Flux2KleinPipeline" if is_klein else "Flux2Pipeline"

    # Step 1: Try from_single_file directly
    if on_message:
        on_message(f"Loading single file ({pipe_name})...")
    try:
        pipe = pipeline_cls.from_single_file(model_path, torch_dtype=dtype, token=hf_token)
        print(f"[flux2] {pipe_cls.__name__}.from_single_file succeeded in {time.time()-t0:.1f}s")
        return pipe
    except Exception as e:
        print(f"[flux2] {pipe_cls.__name__}.from_single_file failed: {e}")
        traceback.print_exc()

    # Step 2: Download only config + VAE + text encoder
    repo = "black-forest-labs/FLUX.2-klein-9B" if is_klein else "black-forest-labs/FLUX.2-dev-9B"

    SKIP_FILES = (
        "flux-2-klein-9b.safetensors",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.fp16.safetensors",
    )

    if on_message:
        on_message(f"Checking cache for {repo}...")

    cached = hf_hub.try_to_load_from_cache(repo_id=repo, filename="model_index.json")
    needs_dl = cached is None or not os.path.exists(cached)

    if needs_dl:
        if on_message:
            on_message("Listing repo files...")

        files = [f for f in hf_hub.list_repo_files(repo, token=hf_token) if f not in SKIP_FILES]
        sizes = {}
        total = 0
        for f in files:
            try:
                s = hf_hub.file_size(repo, f, token=hf_token)
                sizes[f] = s
                total += s
            except Exception:
                pass

        if on_message:
            on_message(f"Downloading components ({total/1024**3:.1f}GB)...")

        dl_base = [0]
        for f in files:
            if cancel_event and cancel_event.is_set():
                print("[flux2] Cancelled during download")
                return None
            for attempt in range(3):
                try:
                    hf_hub.hf_hub_download(repo, f, token=hf_token)
                    dl_base[0] += sizes.get(f, 0)
                    if on_progress:
                        on_progress(dl_base[0], total)
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    if on_message:
                        on_message(f"Retrying {os.path.basename(f)} (attempt {attempt+2})...")

    # Step 3: Load individual components
    if on_message:
        on_message("Loading VAE...")
    vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=dtype, token=hf_token)

    if on_message:
        on_message("Loading text encoder...")
    text_encoder = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype, token=hf_token)

    if on_message:
        on_message("Loading tokenizer...")
    tokenizer = Qwen2TokenizerFast.from_pretrained(repo, subfolder="tokenizer", token=hf_token)

    if on_message:
        on_message("Loading scheduler + transformer config...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler", token=hf_token)
    transformer = Flux2Transformer2DModel.from_config(
        Flux2Transformer2DModel.load_config(repo, subfolder="transformer", token=hf_token),
        torch_dtype=dtype,
    )

    # Step 4: Assemble pipeline
    if on_message:
        on_message("Assembling pipeline...")
    pipe = pipeline_cls(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    pipe.to(dtype=dtype)
    print(f"[flux2] Pipeline assembled in {time.time()-t0:.1f}s")

    if on_message:
        on_message("Swapping checkpoint weights...")

    print(f"[flux2] Reading checkpoint keys via safe_open...")
    unet_state = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        keys = [k for k in f.keys() if k.startswith("model.diffusion_model.")]
        print(f"[flux2] Found {len(keys)} transformer keys, loading...")
        for i, k in enumerate(keys):
            unet_state[k.replace("model.diffusion_model.", "")] = f.get_tensor(k)

    if unet_state:
        t1 = time.time()
        missing, unexpected = pipe.transformer.load_state_dict(unet_state, strict=False)
        print(f"[flux2] Weights swapped in {time.time()-t1:.1f}s. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"[flux2] Missing keys (first 10): {list(missing)[:10]}")
        if unexpected:
            print(f"[flux2] Unexpected keys (first 10): {list(unexpected)[:10]}")

    pipe.to(device=device)
    print(f"[flux2] Pipeline ready in {time.time()-t0:.1f}s total")
    return pipe
