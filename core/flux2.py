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
        print(f"[flux2] {pipeline_cls.__name__}.from_single_file succeeded in {time.time()-t0:.1f}s")
        return pipe
    except Exception as e:
        print(f"[flux2] {pipeline_cls.__name__}.from_single_file failed: {e}")
        traceback.print_exc()

    import importlib.metadata
    print(f"[flux2] diffusers version: {importlib.metadata.version('diffusers')}")
    print(f"[flux2] transformers version: {importlib.metadata.version('transformers')}")

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
    vae_cfg = vae.config
    print(f"[flux2] VAE config: scaling_factor={vae_cfg.get('scaling_factor')}, block_out_channels={vae_cfg.get('block_out_channels')}, latent_channels={vae_cfg.get('latent_channels')}, in_channels={vae_cfg.get('in_channels')}")

    if on_message:
        on_message("Loading text encoder...")
    text_encoder = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype, token=hf_token)

    if on_message:
        on_message("Loading tokenizer...")
    tokenizer = Qwen2TokenizerFast.from_pretrained(repo, subfolder="tokenizer", token=hf_token)

    if on_message:
        on_message("Loading scheduler + transformer config...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler", token=hf_token)

    # Load transformer config and ensure correct architecture
    raw_cfg = Flux2Transformer2DModel.load_config(repo, subfolder="transformer", token=hf_token)
    print(f"[flux2] Transformer raw config: class={raw_cfg.get('_class_name')}, guidance_embeds={raw_cfg.get('guidance_embeds')}, num_layers={raw_cfg.get('num_layers')}, num_single_layers={raw_cfg.get('num_single_layers')}")
    # Remove _class_name to prevent dispatch to wrong model class (e.g. SD3Transformer2DModel)
    raw_cfg.pop("_class_name", None)
    # Klein (distilled) has no guidance embedding
    if is_klein:
        raw_cfg["guidance_embeds"] = False
    transformer = Flux2Transformer2DModel.from_config(raw_cfg, torch_dtype=dtype)

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
        all_ckpt_keys = list(f.keys())
        print(f"[flux2] Total keys in checkpoint: {len(all_ckpt_keys)}")
        # Dump ALL keys grouped by type
        trans_keys = [k for k in all_ckpt_keys if k.startswith("model.diffusion_model.")]
        other_keys = [k for k in all_ckpt_keys if not k.startswith("model.diffusion_model.")]
        print(f"[flux2] model.diffusion_model.* keys: {len(trans_keys)}")
        if other_keys:
            print(f"[flux2] Non-transformer keys ({len(other_keys)}):")
            for k in other_keys:
                print(f"  {k}")
        # Check for VAE keys in checkpoint
        vae_ckpt_keys = [k for k in all_ckpt_keys if "first_stage_model" in k]
        if vae_ckpt_keys:
            print(f"[flux2] WARNING: checkpoint has first_stage_model (VAE) keys: {len(vae_ckpt_keys)}")
        # Check for any key that suggests a different base model
        if any("sdpa" in k or "lora" in k or "diffusion_model." not in k for k in all_ckpt_keys):
            print(f"[flux2] NOTE: checkpoint contains non-standard keys (sdpa/lora/etc) - may be from a different source")
        # Group transformer keys
        double_blocks = sorted([k for k in trans_keys if "double_blocks" in k])
        single_blocks = sorted([k for k in trans_keys if "single_blocks" in k])
        other_trans = sorted([k for k in trans_keys if "double_blocks" not in k and "single_blocks" not in k])
        print(f"[flux2] double_blocks keys: {len(double_blocks)}")
        for k in double_blocks[:5]:
            print(f"  {k}")
        if len(double_blocks) > 5:
            print(f"  ... and {len(double_blocks)-5} more")
        print(f"[flux2] single_blocks keys: {len(single_blocks)}")
        for k in single_blocks[:5]:
            print(f"  {k}")
        if len(single_blocks) > 5:
            print(f"  ... and {len(single_blocks)-5} more")
        if other_trans:
            print(f"[flux2] Other transformer keys ({len(other_trans)}):")
            for k in other_trans:
                print(f"  {k}")
        # Also check if there are transformer_blocks or double_stream keys
        tb_keys = [k for k in all_ckpt_keys if "transformer_blocks" in k]
        ds_keys = [k for k in all_ckpt_keys if "double_stream" in k]
        if tb_keys:
            print(f"[flux2] WARNING: found 'transformer_blocks' keys (diffusers format): {len(tb_keys)}")
        if ds_keys:
            print(f"[flux2] Found 'double_stream' keys: {len(ds_keys)}")

        # Load transformer weights
        print(f"[flux2] Loading {len(trans_keys)} transformer tensors...")
        for i, k in enumerate(trans_keys):
            unet_state[k.replace("model.diffusion_model.", "")] = f.get_tensor(k)

    if unet_state:
        t1 = time.time()
        # Dump all expected model keys for comparison
        model_sd = pipe.transformer.state_dict()
        model_keys = set(model_sd.keys())
        ckpt_keys = set(unet_state.keys())
        print(f"[flux2] Model state_dict total: {len(model_keys)}")
        print(f"[flux2] Checkpoint state_dict total: {len(ckpt_keys)}")
        intersect = model_keys & ckpt_keys
        print(f"[flux2] Matching keys (intersection): {len(intersect)}")
        print(f"[flux2] Keys in model but NOT in checkpoint: {len(model_keys - ckpt_keys)}")
        for k in sorted(model_keys - ckpt_keys)[:20]:
            print(f"  model-only: {k}")
        print(f"[flux2] Keys in checkpoint but NOT in model: {len(ckpt_keys - model_keys)}")
        for k in sorted(ckpt_keys - model_keys)[:20]:
            print(f"  ckpt-only:  {k}")

        missing, unexpected = pipe.transformer.load_state_dict(unet_state, strict=False)
        print(f"[flux2] Weights swapped in {time.time()-t1:.1f}s. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"[flux2] ALL missing keys ({len(missing)}):")
            for k in missing:
                print(f"  missing: {k}")
        if unexpected:
            print(f"[flux2] ALL unexpected keys ({len(unexpected)}):")
            for k in unexpected:
                print(f"  unexpected: {k}")

    pipe.to(device=device)
    print(f"[flux2] Pipeline ready in {time.time()-t0:.1f}s total")
    return pipe
