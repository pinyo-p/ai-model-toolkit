"""Convert FLUX.2-klein single .safetensors checkpoint to diffusers directory format.
Usage: python scripts/convert_flux2_klein.py <input.safetensors> <output_dir> [--repo black-forest-labs/FLUX.2-klein-9B]
"""
import os
import sys
import torch
from safetensors.torch import load_file as safetensors_load_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    repo = sys.argv[4] if len(sys.argv) > 3 and sys.argv[3] == "--repo" else "black-forest-labs/FLUX.2-klein-9B"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"Loading checkpoint: {input_path} ({os.path.getsize(input_path) / 1073741824:.2f} GB)")
    ckpt = safetensors_load_file(input_path, device=device)

    from diffusers import Flux2KleinPipeline
    print(f"Loading base pipeline: {repo}")
    pipe = Flux2KleinPipeline.from_pretrained(repo, torch_dtype=dtype, token=os.environ.get("HF_TOKEN"))

    print("Swapping transformer weights...")
    unet_state = {k.replace("model.diffusion_model.", ""): v for k, v in ckpt.items() if k.startswith("model.diffusion_model.")}
    if unet_state:
        missing, unexpected = pipe.transformer.load_state_dict(unet_state, strict=False)
        print(f"  Transformer: {len(unet_state)} keys loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    del ckpt, unet_state

    print(f"Saving to: {output_dir}")
    pipe.save_pretrained(output_dir, safe_serialization=True)
    print("Done!")
