"""Systematic FLUX.2 Klein debug: isolate checkpoint loading vs pipeline issues."""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_TOKEN"] = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json"))).get("hf_token", "")

from diffusers import Flux2KleinPipeline, AutoencoderKLFlux2, Flux2Transformer2DModel, FlowMatchEulerDiscreteScheduler
from transformers import Qwen3ForCausalLM, Qwen2TokenizerFast
from safetensors import safe_open
from core.flux2 import _remap_flux2_state_dict

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
repo = "black-forest-labs/FLUX.2-klein-9B"
model_path = sys.argv[1] if len(sys.argv) > 1 else "/home/yokiz/stable-diffusion/models/checkpoints/Moody_Desire_Mix.safetensors"

print("="*70)
print("STEP 1: LOAD OFFICIAL TRANSFORMER FROM HF REPO")
print("="*70)

official_transformer = Flux2Transformer2DModel.from_pretrained(
    repo, subfolder="transformer", torch_dtype=dtype, token=os.environ["HF_TOKEN"],
)
official_transformer.to(device, dtype=dtype)
official_transformer.eval()

print(f"  Official transformer loaded: {sum(p.numel() for p in official_transformer.parameters())/1e6:.1f}M params")
print(f"  Config:")
print(f"    num_layers: {official_transformer.config.num_layers}")
print(f"    num_single_layers: {official_transformer.config.num_single_layers}")
print(f"    num_attention_heads: {official_transformer.config.num_attention_heads}")
print(f"    attention_head_dim: {official_transformer.config.attention_head_dim}")
print(f"    joint_attention_dim: {official_transformer.config.joint_attention_dim}")
print(f"    guidance_embeds: {official_transformer.config.guidance_embeds}")
print(f"    mlp_ratio: {official_transformer.config.mlp_ratio}")
print(f"    patch_size: {official_transformer.config.patch_size}")
print(f"    in_channels: {official_transformer.config.in_channels}")
print(f"    out_channels: {official_transformer.config.out_channels}")

official_keys = set(official_transformer.state_dict().keys())
print(f"  Total keys: {len(official_keys)}")

print("\n" + "="*70)
print("STEP 2: LOAD CHECKPOINT + DETAILED REMAP DIAGNOSTICS")
print("="*70)

with safe_open(model_path, framework="pt", device="cpu") as f:
    ckpt_keys = list(f.keys())
    unet_state = {}
    for k in ckpt_keys:
        unet_state[k.replace("model.diffusion_model.", "")] = f.get_tensor(k)

print(f"  Checkpoint total keys: {len(ckpt_keys)}")
print(f"  After stripping prefix: {len(unet_state)}")

# Check checkpoint config from key structure
print(f"\n  Checkpoint key samples:")
for k in sorted(unet_state.keys())[:10]:
    print(f"    {k}: {unet_state[k].shape}")
print(f"    ... and {len(unet_state)-10} more")

# Check dimension hints from checkpoint
for hint_key in ["img_in.weight", "txt_in.weight", "time_in.in_layer.weight", "final_layer.linear.weight"]:
    if hint_key in unet_state:
        print(f"  {hint_key}: {unet_state[hint_key].shape}")

# Check double block structure
db0_keys = sorted([k for k in unet_state.keys() if k.startswith("double_blocks.0.")])
print(f"\n  double_blocks.0 keys ({len(db0_keys)}):")
for k in db0_keys:
    print(f"    {k}: {unet_state[k].shape}")

# Check single block structure
sb0_keys = sorted([k for k in unet_state.keys() if k.startswith("single_blocks.0.")])
print(f"\n  single_blocks.0 keys ({len(sb0_keys)}):")
for k in sb0_keys:
    print(f"    {k}: {unet_state[k].shape}")

# Now create model with same config and do remap
custom_transformer = Flux2Transformer2DModel.from_config(
    official_transformer.config, torch_dtype=dtype,
)
custom_transformer.to(device, dtype=dtype)
custom_transformer.eval()

model_sd = custom_transformer.state_dict()
remapped = _remap_flux2_state_dict(unet_state, model_sd)

print(f"\n  Remap results:")
print(f"    Keys produced by remap: {len(remapped)}")
print(f"    Keys expected by model: {len(model_sd)}")

# Detailed coverage check
matched = 0
shape_mismatch = []
for df_key, tensor in sorted(remapped.items()):
    if df_key in model_sd:
        if tensor.shape == model_sd[df_key].shape:
            matched += 1
        else:
            shape_mismatch.append(f"    {df_key}: ckpt={tensor.shape} vs model={model_sd[df_key].shape}")
    else:
        print(f"    UNEXPECTED key: {df_key} ({tensor.shape})")

missing_after_load = [k for k in model_sd.keys() if k not in remapped]

print(f"    Shape-matched: {matched}/{len(model_sd)}")
print(f"    Coverage: {matched/len(model_sd):.2%}")
if shape_mismatch:
    print(f"    Shape MISMATCHES ({len(shape_mismatch)}):")
    for m in shape_mismatch:
        print(m)
if missing_after_load:
    print(f"    Missing after remap ({len(missing_after_load)}):")
    for k in missing_after_load:
        print(f"      {k}: {model_sd[k].shape}")

# Load weights
missing_keys, unexpected_keys = custom_transformer.load_state_dict(remapped, strict=False)
print(f"\n  load_state_dict(strict=False):")
print(f"    Missing: {len(missing_keys)}")
print(f"    Unexpected: {len(unexpected_keys)}")
if missing_keys:
    for k in missing_keys:
        print(f"      {k}")

# Check specific critical key groups
print(f"\n  Key group coverage:")
groups = {
    "x_embedder": [k for k in remapped if "x_embedder" in k],
    "context_embedder": [k for k in remapped if "context_embedder" in k],
    "time_guidance_embed": [k for k in remapped if "time_guidance_embed" in k],
    "proj_out": [k for k in remapped if "proj_out" in k],
    "norm_out": [k for k in remapped if "norm_out" in k],
    "transformer_blocks (double)": [k for k in remapped if "transformer_blocks" in k],
    "single_transformer_blocks": [k for k in remapped if "single_transformer_blocks" in k],
    "double_stream_modulation": [k for k in remapped if "double_stream_modulation" in k],
    "single_stream_modulation": [k for k in remapped if "single_stream_modulation" in k],
}
for gname, gkeys in groups.items():
    expected = [k for k in model_sd if gname.split(" (")[0].replace("_", ".").replace("double.stream", "double_stream").replace("single.stream", "single_stream") in k.replace("_", ".")]
    # Simpler: count from model_sd
    expected_count = len([k for k in model_sd if gname.split(" (")[0] in k])
    print(f"    {gname}: {len(gkeys)}/{expected_count} mapped")

print("\n" + "="*70)
print("STEP 3: COMPARE OFFICIAL vs CUSTOM WEIGHTS")
print("="*70)

official_sd = official_transformer.state_dict()
custom_sd = custom_transformer.state_dict()

print(f"  Official keys: {len(official_sd)}")
print(f"  Custom keys:   {len(custom_sd)}")

# Check if key sets match
official_only = set(official_sd.keys()) - set(custom_sd.keys())
custom_only = set(custom_sd.keys()) - set(official_sd.keys())
print(f"  Keys only in official: {len(official_only)}")
print(f"  Keys only in custom:   {len(custom_only)}")

if official_only:
    for k in sorted(official_only)[:10]:
        print(f"    {k}")

# For matching keys, compare weight stats
weight_diffs = []
for key in sorted(set(official_sd.keys()) & set(custom_sd.keys())):
    o = official_sd[key].float()
    c = custom_sd[key].float()
    if o.shape == c.shape:
        cos_sim = torch.nn.functional.cosine_similarity(o.flatten().unsqueeze(0), c.flatten().unsqueeze(0)).item()
        max_diff = (o - c).abs().max().item()
        if cos_sim < 0.99:
            weight_diffs.append((key, cos_sim, max_diff, o.shape))

if weight_diffs:
    print(f"\n  Keys with cosine similarity < 0.99 ({len(weight_diffs)}):")
    for k, cos, md, shape in sorted(weight_diffs, key=lambda x: x[1])[:20]:
        print(f"    {k}: cos={cos:.6f}, max_diff={md:.6f}, shape={shape}")
else:
    print(f"\n  All {len(set(official_sd.keys()) & set(custom_sd.keys()))} shared keys have cosine similarity >= 0.99 ✓")

print("\n" + "="*70)
print("STEP 4: GENERATE WITH OFFICIAL TRANSFORMER (baseline)")
print("="*70)

# Build pipeline with OFFICIAL transformer
scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False,
)
vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=dtype, token=os.environ["HF_TOKEN"])
text_encoder = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype, token=os.environ["HF_TOKEN"])
tokenizer = Qwen2TokenizerFast.from_pretrained(repo, subfolder="tokenizer", token=os.environ["HF_TOKEN"])

pipe_official = Flux2KleinPipeline(
    scheduler=scheduler, vae=vae, text_encoder=text_encoder,
    tokenizer=tokenizer, transformer=official_transformer,
)
pipe_official.to(device, dtype=dtype)
pipe_official.vae.to(dtype=torch.float32)
_orig_decode_off = pipe_official.vae.decode
pipe_official.vae.decode = lambda z, *a, **kw: _orig_decode_off(z.to(torch.float32), *a, **kw)

prompt = "a cat, high quality"
gen = torch.Generator(device=device).manual_seed(42)

print("  Running official transformer...")
img_official = pipe_official(prompt=prompt, num_inference_steps=4, guidance_scale=1.0, generator=gen).images[0]
img_official.save("/tmp/debug_official.png")
print("  Saved /tmp/debug_official.png")

print("\n" + "="*70)
print("STEP 5: GENERATE WITH CUSTOM CHECKPOINT TRANSFORMER")
print("="*70)

# Build pipeline with CUSTOM transformer
scheduler2 = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000, shift=1.0, use_dynamic_shifting=False,
)
vae2 = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=dtype, token=os.environ["HF_TOKEN"])
text_encoder2 = Qwen3ForCausalLM.from_pretrained(repo, subfolder="text_encoder", torch_dtype=dtype, token=os.environ["HF_TOKEN"])

pipe_custom = Flux2KleinPipeline(
    scheduler=scheduler2, vae=vae2, text_encoder=text_encoder2,
    tokenizer=tokenizer, transformer=custom_transformer,
)
pipe_custom.to(device, dtype=dtype)
pipe_custom.vae.to(dtype=torch.float32)
_orig_decode_custom = pipe_custom.vae.decode
pipe_custom.vae.decode = lambda z, *a, **kw: _orig_decode_custom(z.to(torch.float32), *a, **kw)

gen2 = torch.Generator(device=device).manual_seed(42)
print("  Running custom checkpoint transformer...")
img_custom = pipe_custom(prompt=prompt, num_inference_steps=4, guidance_scale=1.0, generator=gen2).images[0]
img_custom.save("/tmp/debug_custom.png")
print("  Saved /tmp/debug_custom.png")

print("\n" + "="*70)
print("COMPARISON DONE")
print("="*70)
print("  /tmp/debug_official.png = official HF transformer")
print("  /tmp/debug_custom.png   = Moody_Desire_Mix checkpoint")
print("  Compare the two images to determine if issue is in:")
print("    - checkpoint weights (if custom looks wrong)")
print("    - pipeline/scheduler (if both look wrong)")
print("    - config (if shapes mismatch)")
