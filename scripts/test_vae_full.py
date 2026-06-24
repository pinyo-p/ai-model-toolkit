"""Test the FULL VAE encode/decode including patchify+bn steps (pipeline path)."""
import os, sys, json, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "HF_TOKEN" not in os.environ:
    sf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "settings.json")
    if os.path.exists(sf):
        s = json.load(open(sf))
        os.environ["HF_TOKEN"] = s.get("hf_token", "")
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.huggingface/token")).read().strip()

import numpy as np
from PIL import Image
from diffusers import AutoencoderKLFlux2
import torch.nn.functional as F
from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
repo = "black-forest-labs/FLUX.2-klein-9B"

vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=dtype, token=os.environ["HF_TOKEN"])
vae.to(device)
vae.eval()

def _patchify_latents(latents):
    B, C, H, W = latents.shape
    latents = latents.view(B, C, H // 2, 2, W // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(B, C * 4, H // 2, W // 2)
    return latents

def _unpatchify_latents(latents):
    B, C, H, W = latents.shape
    latents = latents.reshape(B, C // 4, 2, 2, H, W)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(B, C // 4, H * 2, W * 2)
    return latents

def _pack_latents(latents):
    B, C, H, W = latents.shape
    return latents.reshape(B, C, H * W).permute(0, 2, 1)

# Load a test image
out_dir = "/home/yokiz/ai-model-toolkit/output"
test_img = None
for f in sorted(os.listdir(out_dir)):
    if f.endswith(".png"):
        test_img = os.path.join(out_dir, f)
        break
if not test_img:
    print("[test] No output images found, using dummy input")
    test_tensor = torch.randn(1, 3, 1024, 1024, device=device, dtype=dtype)
else:
    print(f"[test] Loading test image: {test_img}")
    img = Image.open(test_img).convert("RGB").resize((1024, 1024))
    test_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
    test_tensor = test_tensor.to(dtype)

# Full pipeline VAE path:
#   Encode:  patchify → (latent - bn_mean) / bn_std → pack (this is what transformer sees)
#   Decode:  unpack → latent * bn_std + bn_mean → unpatchify → vae.decode
with torch.no_grad():
    # === Encode path (matches pipeline's _encode_vae_image) ===
    posterior = vae.encode(test_tensor).latent_dist
    raw_latent = posterior.sample()  # (1, 32, 128, 128)
    print(f"[test] Raw latent: {raw_latent.shape}, mean={raw_latent.mean():.4f}, std={raw_latent.std():.4f}")

    patched = _patchify_latents(raw_latent)  # (1, 128, 64, 64)
    print(f"[test] Patched: {patched.shape}, mean={patched.mean():.4f}, std={patched.std():.4f}")

    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(patched.device, patched.dtype)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1).to(patched.device, patched.dtype) + vae.config.batch_norm_eps)
    print(f"[test] bn_mean range: [{bn_mean.min():.4f}, {bn_mean.max():.4f}]")
    print(f"[test] bn_std range: [{bn_std.min():.4f}, {bn_std.max():.4f}]")

    # Normalize for transformer (what the transformer outputs should look like)
    normed = (patched - bn_mean) / bn_std
    print(f"[test] After bn norm (transformer input): mean={normed.mean():.4f}, std={normed.std():.4f}")
    print(f"[test] normed range: [{normed.min():.4f}, {normed.max():.4f}]")

    # === Decode path (matches pipeline decode) ===
    # This is what happens after denoising: take the (allegedly) normalized latents,
    # denormalize, unpatchify, decode
    denormed = normed * bn_std + bn_mean
    print(f"[test] After bn denorm: mean={denormed.mean():.4f}, std={denormed.std():.4f}")

    unpatched = _unpatchify_latents(denormed)  # (1, 32, 128, 128)
    print(f"[test] Unpatched: {unpatched.shape}, mean={unpatched.mean():.4f}, std={unpatched.std():.4f}")

    # VAE decode (matches vae.decode path)
    decoded = vae.decode(unpatched).sample
    print(f"[test] Decoded: {decoded.shape}")

    mse = F.mse_loss(decoded, test_tensor)
    print(f"[test] Full pipeline reconstruction MSE: {mse.item():.6f}")

    # Also show what happens if we skip the bn denorm (i.e., transformer outputs raw latents)
    # This is what CURRENTLY happens if transformer weights are wrong
    print(f"\n[test] What if we decode WITHOUT bn denorm (raw latents directly):")
    decoded_raw = vae.decode(_unpatchify_latents(patched)).sample
    mse_raw = F.mse_loss(decoded_raw, test_tensor)
    print(f"[test]   MSE: {mse_raw.item():.6f}")

    decoded_np = decoded[0].float().cpu().numpy()
    decoded_img = ((decoded_np.clip(-1,1) + 1) * 127.5).astype(np.uint8).transpose(1,2,0)
    Image.fromarray(decoded_img).save("/tmp/vae_full_recon.png")
    print(f"[test] Saved to /tmp/vae_full_recon.png")

print("[test] Done")
