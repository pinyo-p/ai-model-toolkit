"""Debug script: run ONE generation step and dump intermediate values."""
import os, sys, json, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_TOKEN"] = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json"))).get("hf_token", "")

import numpy as np
from PIL import Image
from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
import huggingface_hub as hf_hub

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
repo = "black-forest-labs/FLUX.2-klein-9B"

model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/yokiz/stable-diffusion/models/checkpoints/Moody_Desire_Mix.safetensors"

name_lower = os.path.basename(model_path).lower()
is_klein = "klein" in name_lower or "schnell" in name_lower

# Load components (same as flux2.py)
vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=dtype, token=os.environ["HF_TOKEN"])
vae.to(device)
text_encoder = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype, token=os.environ["HF_TOKEN"])
text_encoder.to(device)
tokenizer = Qwen2TokenizerFast.from_pretrained(repo, subfolder="tokenizer", token=os.environ["HF_TOKEN"])
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(repo, subfolder="scheduler", token=os.environ["HF_TOKEN"])

raw_cfg = Flux2Transformer2DModel.load_config(repo, subfolder="transformer", token=os.environ["HF_TOKEN"])
raw_cfg.pop("_class_name", None)
if is_klein:
    raw_cfg["guidance_embeds"] = False
transformer = Flux2Transformer2DModel.from_config(raw_cfg, torch_dtype=dtype)

# Load checkpoint weights
from safetensors import safe_open
from core.flux2 import _remap_flux2_state_dict

t0 = torch.cuda.Event(enable_timing=True)
t1 = torch.cuda.Event(enable_timing=True)
t0.record()

with safe_open(model_path, framework="pt", device="cpu") as f:
    unet_state = {}
    for k in f.keys():
        unet_state[k.replace("model.diffusion_model.", "")] = f.get_tensor(k)

model_sd = transformer.state_dict()
remapped = _remap_flux2_state_dict(unet_state, model_sd)
missing, unexpected = transformer.load_state_dict(remapped, strict=False)
print(f"[debug] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if unexpected:
    print(f"[debug] Unexpected keys:")
    for k in unexpected:
        print(f"  {k}")

transformer.to(device)
transformer.eval()

t1.record()
torch.cuda.synchronize()
print(f"[debug] Model loaded in {t0.elapsed_time(t1)/1000:.1f}s")

# Assemble pipeline
pipe = Flux2KleinPipeline(
    scheduler=scheduler, vae=vae, text_encoder=text_encoder,
    tokenizer=tokenizer, transformer=transformer,
)
pipe.to(device)

# Run generation with callback to capture latents
prompt = "a cat, high quality"
print(f"[debug] Generating with prompt: {prompt}")

captured_latents = []

def step_cb(pipeline, step_index, timestep, callback_kwargs):
    latents = callback_kwargs["latents"]
    captured_latents.append({
        "step": step_index,
        "latents_mean": latents.mean().item(),
        "latents_std": latents.std().item(),
        "latents_min": latents.min().item(),
        "latents_max": latents.max().item(),
    })
    return callback_kwargs

image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device=device).manual_seed(42),
    callback_on_step_end=step_cb,
    callback_on_step_end_tensor_inputs=["latents"],
).images[0]

image.save("/tmp/debug_output.png")
print(f"[debug] Saved /tmp/debug_output.png")

print(f"[debug] Latent evolution:")
for c in captured_latents:
    print(f"  Step {c['step']}: mean={c['latents_mean']:.4f}, std={c['latents_std']:.4f}, range=[{c['latents_min']:.4f}, {c['latents_max']:.4f}]")

print("[debug] Done")
