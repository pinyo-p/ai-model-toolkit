"""Test AutoencoderKLFlux2 by encoding & decoding a real image."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["HF_TOKEN"] = open(os.path.expanduser("~/.huggingface/token")).read().strip()

import numpy as np
from PIL import Image
from diffusers import AutoencoderKLFlux2
import torch.nn.functional as F

dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
repo = "black-forest-labs/FLUX.2-klein-9B"

# Load VAE
vae = AutoencoderKLFlux2.from_pretrained(repo, subfolder="vae", torch_dtype=dtype, token=os.environ["HF_TOKEN"])
vae.to(device)
vae.eval()
print(f"[vae] Config: scaling_factor={vae.config.scaling_factor}, latent_channels={vae.config.latent_channels}")
print(f"[vae] block_out_channels={vae.config.block_out_channels}")

# Load a test image (first .png from output dir)
out_dir = "/home/yokiz/ai-model-toolkit/output"
test_img = None
for f in sorted(os.listdir(out_dir)):
    if f.endswith(".png"):
        test_img = os.path.join(out_dir, f)
        break
if not test_img:
    # Generate a dummy image
    print("[vae] No output images found, using dummy input")
    test_tensor = torch.randn(1, 3, 1024, 1024, device=device, dtype=dtype)
else:
    print(f"[vae] Loading test image: {test_img}")
    img = Image.open(test_img).convert("RGB").resize((1024, 1024))
    test_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).unsqueeze(0).float().to(device) / 127.5 - 1.0
    test_tensor = test_tensor.to(dtype)

# Encode
with torch.no_grad():
    print(f"[vae] Encoding ({test_tensor.shape})...")
    posterior = vae.encode(test_tensor).latent_dist
    latent = posterior.sample()
    print(f"[vae] Latent shape: {latent.shape}, dtype: {latent.dtype}")
    print(f"[vae] Latent stats: mean={latent.mean().item():.4f}, std={latent.std().item():.4f}, min={latent.min().item():.4f}, max={latent.max().item():.4f}")

    # Decode
    print(f"[vae] Decoding...")
    decoded = vae.decode(latent).sample
    print(f"[vae] Decoded shape: {decoded.shape}, dtype: {decoded.dtype}")
    print(f"[vae] Decoded stats: mean={decoded.mean().item():.4f}, std={decoded.std().item():.4f}, min={decoded.min().item():.4f}, max={decoded.max().item():.4f}")

    # Reconstruction error
    mse = F.mse_loss(decoded, test_tensor)
    print(f"[vae] Reconstruction MSE: {mse.item():.6f}")

    # Save latent + decoded
    latent_np = latent[0].float().cpu().numpy()
    decoded_np = decoded[0].float().cpu().numpy()
    decoded_img = ((decoded_np.clip(-1,1) + 1) * 127.5).astype(np.uint8).transpose(1,2,0)
    Image.fromarray(decoded_img).save("/tmp/vae_recon_test.png")
    print(f"[vae] Reconstructed image saved: /tmp/vae_recon_test.png")

    # Print per-channel latent stats
    for c in range(latent.shape[1]):
        ch = latent[0,c]
        print(f"  ch[{c:2d}]: mean={ch.mean().item():+.4f} std={ch.std().item():.4f} min={ch.min().item():+.4f} max={ch.max().item():+.4f}")

print("[vae] Done")
