import torch
import os
from diffusers import ZImagePipeline, AutoencoderKL
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_zimage_pipeline(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    vae_path: str = None,
    text_encoder_path: str = None,
    local_vae: torch.nn.Module = None,
    on_message=None,
):
    from diffusers import ZImagePipeline
    hf_token = os.environ.get("HF_TOKEN")
    zimage_repo = "Tongyi-MAI/Z-Image-Turbo"

    # HF repo ID (not local) -> use from_pretrained directly
    if not os.path.isfile(model_path) and not os.path.isdir(model_path):
        if on_message:
            on_message("Loading Z-Image-Turbo from Hugging Face...")
        return ZImagePipeline.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=False, token=hf_token
        )

    # Local directory (diffusers format)
    if os.path.isdir(model_path):
        if on_message:
            on_message("Loading Z-Image-Turbo from local directory...")
        return ZImagePipeline.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=False, token=hf_token
        )

    # Local single file
    if on_message:
        on_message("Loading Z-Image-Turbo from single file...")

    # First try: from_single_file without VAE (works if checkpoint has VAE embedded)
    try:
        return ZImagePipeline.from_single_file(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=False, token=hf_token
        )
    except Exception as e:
        if "AutoencoderKL" not in str(e):
            raise

    # VAE is missing in checkpoint - load VAE from HF repo
    if on_message:
        on_message("VAE not found in checkpoint, loading from Hugging Face...")

    vae = local_vae
    if vae is None and vae_path and os.path.exists(vae_path):
        try:
            if os.path.isfile(vae_path) and vae_path.endswith('.safetensors'):
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
            else:
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        except Exception as ve:
            vae = None
            if on_message:
                on_message(f"VAE local load failed: {ve}")

    if vae is None:
        try:
            vae = AutoencoderKL.from_pretrained(
                zimage_repo, subfolder="vae", torch_dtype=dtype, token=hf_token
            )
        except Exception as ve:
            if on_message:
                on_message(f"HF VAE load failed: {ve}")
            vae = None

    if vae is None:
        raise RuntimeError(
            "Failed to load AutoencoderKL for Z-Image.\n"
            "Please download VAE manually from: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/vae\n"
            "Then set the VAE path in settings."
        )

    # Load text_encoder + tokenizer from HF repo (only if needed)
    text_encoder = None
    tokenizer = None
    model_dir = os.path.dirname(model_path)

    # Try local text_encoder first
    local_te_paths = [
        text_encoder_path,
        os.path.join(model_dir, "text_encoder"),
        os.path.join(model_dir, "phi"),
    ]
    for tp in local_te_paths:
        if tp and os.path.exists(tp):
            try:
                if os.path.isfile(tp) and tp.endswith('.safetensors'):
                    text_encoder = AutoModelForCausalLM.from_single_file(tp, torch_dtype=dtype)
                else:
                    text_encoder = AutoModelForCausalLM.from_pretrained(tp, torch_dtype=dtype)
                break
            except Exception:
                pass

    # Try local tokenizer
    local_tok_paths = [
        os.path.join(model_dir, "tokenizer"),
        text_encoder_path,
        os.path.join(model_dir, "text_encoder"),
    ]
    for tok_p in local_tok_paths:
        if tok_p and os.path.exists(tok_p):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_p, trust_remote_code=True)
                if getattr(tokenizer, 'chat_template', None):
                    break
            except Exception:
                pass

    # Fallback: download from HF repo
    if text_encoder is None:
        if on_message:
            on_message("Loading text_encoder from Hugging Face...")
        try:
            text_encoder = AutoModelForCausalLM.from_pretrained(
                zimage_repo, subfolder="text_encoder", torch_dtype=dtype, token=hf_token
            )
        except Exception as te:
            if on_message:
                on_message(f"HF text_encoder load failed: {te}")
    if tokenizer is None or not getattr(tokenizer, 'chat_template', None):
        if on_message:
            on_message("Loading tokenizer from Hugging Face...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                zimage_repo, subfolder="tokenizer", trust_remote_code=True, token=hf_token
            )
        except Exception as toke:
            if on_message:
                on_message(f"HF tokenizer load failed: {toke}")

    # Try from_single_file with VAE + optional components
    kwargs = dict(torch_dtype=dtype, vae=vae)
    if text_encoder is not None:
        kwargs['text_encoder'] = text_encoder
    if tokenizer is not None:
        kwargs['tokenizer'] = tokenizer
    kwargs['low_cpu_mem_usage'] = False

    if on_message:
        on_message("Loading Z-Image with VAE...")
    return ZImagePipeline.from_single_file(model_path, **kwargs)
