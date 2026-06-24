"""Debug script: run ONE generation and dump ALL intermediate values to diagnose denoising."""
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

# Load components
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

with safe_open(model_path, framework="pt", device="cpu") as f:
    unet_state = {}
    for k in f.keys():
        unet_state[k.replace("model.diffusion_model.", "")] = f.get_tensor(k)

model_sd = transformer.state_dict()
remapped = _remap_flux2_state_dict(unet_state, model_sd)
missing, unexpected = transformer.load_state_dict(remapped, strict=False)
print(f"[debug] Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    for k in missing:
        print(f"  missing: {k}")
if unexpected:
    for k in unexpected:
        print(f"  unexpected: {k}")

transformer.to(device, dtype=dtype)
transformer.eval()

# ======== SCHEDULER DIAGNOSTICS ========
print("\n" + "="*60)
print("SCHEDULER CONFIG")
print("="*60)
print(f"  shift: {scheduler.config.shift}")
print(f"  use_dynamic_shifting: {scheduler.config.use_dynamic_shifting}")
print(f"  scheduler class: {scheduler.__class__.__name__}")
print(f"  num_train_timesteps: {scheduler.config.num_train_timesteps}")
print(f"  use_karras_sigmas: {getattr(scheduler.config, 'use_karras_sigmas', 'N/A')}")
print(f"  use_lucretine_sigmas: {getattr(scheduler.config, 'use_lucretine_sigmas', 'N/A')}")

# ======== TRANSFORMER DIAGNOSTICS ========
print("\n" + "="*60)
print("TRANSFORMER CONFIG")
print("="*60)
print(f"  inner_dim: {transformer.config.inner_dim}")
print(f"  num_heads: {transformer.config.num_heads}")
print(f"  head_dim: {transformer.config.head_dim}")
print(f"  num_layers: {transformer.config.num_layers}")
print(f"  num_single_layers: {transformer.config.num_single_layers}")
print(f"  guidance_embeds: {transformer.config.guidance_embeds}")
print(f"  mlp_ratio: {transformer.config.mlp_ratio}")

# ======== MONKEY-PATCH TRANSFORMER TO CAPTURE OUTPUT ========
orig_transformer_forward = transformer.forward
model_outputs = []
model_inputs = []

def patched_transformer_forward(hidden_states, timestep, encoder_hidden_states, guidance=None, *args, **kwargs):
    # Capture input stats
    model_inputs.append({
        'hidden_states_mean': hidden_states.float().mean().item(),
        'hidden_states_std': hidden_states.float().std().item(),
        'timestep_val': timestep.item() if hasattr(timestep, 'item') else float(timestep),
        'encoder_mean': encoder_hidden_states.float().mean().item(),
        'encoder_std': encoder_hidden_states.float().std().item(),
    })
    # Run original forward
    result = orig_transformer_forward(hidden_states, timestep, encoder_hidden_states, guidance, *args, **kwargs)
    # Capture output stats
    out = result.sample if hasattr(result, 'sample') else result[0]
    model_outputs.append({
        'output_mean': out.float().mean().item(),
        'output_std': out.float().std().item(),
        'output_min': out.float().min().item(),
        'output_max': out.float().max().item(),
    })
    return result

transformer.forward = patched_transformer_forward

# ======== MONKEY-PATCH SCHEDULER STEP ========
orig_scheduler_step = scheduler.step
scheduler_steps = []

def patched_scheduler_step(model_output, timestep, sample, *args, **kwargs):
    result = orig_scheduler_step(model_output, timestep, sample, *args, **kwargs)
    prev = result.prev_sample
    scheduler_steps.append({
        'timestep': timestep.item() if hasattr(timestep, 'item') else float(timestep),
        'sample_before_mean': sample.float().mean().item(),
        'sample_before_std': sample.float().std().item(),
        'prev_sample_mean': prev.float().mean().item(),
        'prev_sample_std': prev.float().std().item(),
        'prev_sample_min': prev.float().min().item(),
        'prev_sample_max': prev.float().max().item(),
    })
    return result

scheduler.step = patched_scheduler_step

# ======== ASSEMBLE PIPELINE ========
pipe = Flux2KleinPipeline(
    scheduler=scheduler, vae=vae, text_encoder=text_encoder,
    tokenizer=tokenizer, transformer=transformer,
)
pipe.to(device, dtype=dtype)

# VAE precision fix
pipe.vae.to(dtype=torch.float32)
orig_vae_decode = pipe.vae.decode
pipe.vae.decode = lambda z, *a, **kw: orig_vae_decode(z.to(torch.float32), *a, **kw)

# ======== RUN GENERATION ========
prompt = "a cat, high quality"
print(f"\n[debug] Generating: '{prompt}' (steps=4, cfg=1.0, seed=42)")

image = pipe(
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]

image.save("/tmp/debug_output.png")
print(f"[debug] Saved /tmp/debug_output.png")

# ======== DUMP ALL CAPTURED DATA ========
print("\n" + "="*60)
print("SCHEDULER TIMESTEPS (computed by scheduler.set_timesteps)")
print("="*60)
print(f"  timesteps: {scheduler.timesteps.tolist()}")
print(f"  sigmas: {scheduler.sigmas.tolist()}")

print("\n" + "="*60)
print("TRANSFORMER INPUT/OUTPUT PER STEP")
print("="*60)
for i, (inp, out) in enumerate(zip(model_inputs, model_outputs)):
    print(f"\n  Step {i}:")
    print(f"    INPUT  hidden_states: mean={inp['hidden_states_mean']:.6f}, std={inp['hidden_states_std']:.6f}")
    print(f"    INPUT  timestep:     {inp['timestep_val']:.6f}")
    print(f"    INPUT  encoder:      mean={inp['encoder_mean']:.6f}, std={inp['encoder_std']:.6f}")
    print(f"    OUTPUT sample:       mean={out['output_mean']:.6f}, std={out['output_std']:.6f}")
    print(f"    OUTPUT range:        [{out['output_min']:.6f}, {out['output_max']:.6f}]")

print("\n" + "="*60)
print("SCHEDULER STEP RESULTS PER STEP")
print("="*60)
for i, s in enumerate(scheduler_steps):
    delta_std = s['prev_sample_std'] - s['sample_before_std']
    print(f"\n  Step {i}:")
    print(f"    timestep:       {s['timestep']:.6f}")
    print(f"    BEFORE sample:  mean={s['sample_before_mean']:.6f}, std={s['sample_before_std']:.6f}")
    print(f"    AFTER sample:   mean={s['prev_sample_mean']:.6f}, std={s['prev_sample_std']:.6f}")
    print(f"    AFTER range:    [{s['prev_sample_min']:.6f}, {s['prev_sample_max']:.6f}]")
    print(f"    delta std:      {delta_std:+.6f} ({'REDUCED' if delta_std < 0 else 'INCREASED'})")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if scheduler_steps:
    first_std = scheduler_steps[0]['sample_before_std']
    last_std = scheduler_steps[-1]['prev_sample_std']
    print(f"  Initial noise std:  {first_std:.6f}")
    print(f"  Final latent std:   {last_std:.6f}")
    print(f"  Change:             {last_std - first_std:+.6f}")
    print(f"  Direction:          {'CORRECT (denoising)' if last_std < first_std else 'WRONG (noise increasing!)'}")

print("\n[debug] Done")
