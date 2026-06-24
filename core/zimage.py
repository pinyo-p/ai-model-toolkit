import torch
import os
from diffusers import ZImagePipeline, AutoencoderKL
from transformers import AutoModelForCausalLM, AutoTokenizer


def _try_load_vae(vae_path, local_vae, dtype, zimage_repo, hf_token):
    vae = local_vae
    if vae is None and vae_path and os.path.exists(vae_path):
        try:
            if os.path.isfile(vae_path) and vae_path.endswith('.safetensors'):
                vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
            else:
                vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
        except Exception:
            vae = None
    if vae is None:
        try:
            vae = AutoencoderKL.from_pretrained(
                zimage_repo, subfolder="vae", torch_dtype=dtype, token=hf_token
            )
        except Exception:
            vae = None
    return vae


def _try_load_text_encoder(text_encoder_path, model_dir, dtype, zimage_repo, hf_token):
    paths = [
        text_encoder_path,
        os.path.join(model_dir, "text_encoder"),
        os.path.join(model_dir, "phi"),
    ]
    for tp in paths:
        if tp and os.path.exists(tp):
            try:
                if os.path.isfile(tp) and tp.endswith('.safetensors'):
                    return AutoModelForCausalLM.from_single_file(tp, torch_dtype=dtype)
                return AutoModelForCausalLM.from_pretrained(tp, torch_dtype=dtype)
            except Exception:
                pass
    try:
        return AutoModelForCausalLM.from_pretrained(
            zimage_repo, subfolder="text_encoder", torch_dtype=dtype, token=hf_token
        )
    except Exception:
        return None


def _try_load_tokenizer(text_encoder_path, model_dir, dtype, zimage_repo, hf_token):
    paths = [
        os.path.join(model_dir, "tokenizer"),
        text_encoder_path,
        os.path.join(model_dir, "text_encoder"),
    ]
    for tp in paths:
        if tp and os.path.exists(tp):
            try:
                tok = AutoTokenizer.from_pretrained(tp, trust_remote_code=True)
                if getattr(tok, 'chat_template', None):
                    return tok
            except Exception:
                pass
    try:
        return AutoTokenizer.from_pretrained(
            zimage_repo, subfolder="tokenizer", trust_remote_code=True, token=hf_token
        )
    except Exception:
        return None


def load_zimage_pipeline(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    vae_path: str = None,
    text_encoder_path: str = None,
    local_vae: torch.nn.Module = None,
    on_message=None,
):
    hf_token = os.environ.get("HF_TOKEN")
    zimage_repo = "Tongyi-MAI/Z-Image-Turbo"

    if not os.path.isfile(model_path) and not os.path.isdir(model_path):
        if on_message:
            on_message("Loading Z-Image-Turbo from Hugging Face...")
        return ZImagePipeline.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=False, token=hf_token
        )

    if os.path.isdir(model_path):
        if on_message:
            on_message("Loading Z-Image-Turbo from local directory...")
        return ZImagePipeline.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=False, token=hf_token
        )

    if on_message:
        on_message("Loading Z-Image-Turbo from single file...")

    model_dir = os.path.dirname(model_path)

    if on_message:
        on_message("Loading VAE...")
    vae = _try_load_vae(vae_path, local_vae, dtype, zimage_repo, hf_token)
    if vae is None:
        raise RuntimeError(
            "Failed to load AutoencoderKL for Z-Image.\n"
            "Please download VAE manually from: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/vae\n"
            "Then set the VAE path in settings."
        )

    if on_message:
        on_message("Loading text encoder...")
    text_encoder = _try_load_text_encoder(text_encoder_path, model_dir, dtype, zimage_repo, hf_token)
    if text_encoder is None:
        raise RuntimeError(
            "Failed to load text encoder for Z-Image.\n"
            "Please download text_encoder manually from: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/text_encoder\n"
            "Then set the text_encoder path in settings."
        )

    if on_message:
        on_message("Loading tokenizer...")
    tokenizer = _try_load_tokenizer(text_encoder_path, model_dir, dtype, zimage_repo, hf_token)

    kwargs = dict(
        torch_dtype=dtype, vae=vae, text_encoder=text_encoder,
        low_cpu_mem_usage=False,
    )
    if tokenizer is not None:
        kwargs['tokenizer'] = tokenizer

    if on_message:
        on_message("Loading Z-Image from single file (VAE + text encoder injected)...")
    return ZImagePipeline.from_single_file(model_path, **kwargs)
