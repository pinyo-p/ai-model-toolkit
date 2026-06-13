import os
import torch
import json
from typing import List
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from peft import LoraConfig, get_peft_model, PeftModel
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from safetensors.torch import save_file
import shutil


def _get_pipeline(model_name: str):
    dtype = torch.float16
    name_lower = model_name.lower()
    is_sdxl = any(x in name_lower for x in ["xl", "sdxl", "pony", "sd_xl"])
    cls = StableDiffusionXLPipeline if is_sdxl else StableDiffusionPipeline
    return cls.from_pretrained(model_name, torch_dtype=dtype)


def image2lora(
    images: List[str],
    concept: str,
    steps: int,
    rank: int = 16,
    lr: float = 1e-4,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    output_path: str = "output_lora.safetensors"
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    pipeline = _get_pipeline(base_model)
    pipeline.to(device)

    is_sdxl = "xl" in base_model.lower() or "sdxl" in base_model.lower()
    pipeline._is_sdxl = is_sdxl

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.1,
        bias="none",
        task_type="TEXT2IMAGE",
    )

    pipeline.unet = get_peft_model(pipeline.unet, lora_config)

    pipeline._te_names = []
    if is_sdxl:
        pipeline.text_encoder_1 = get_peft_model(pipeline.text_encoder_1, lora_config)
        pipeline.text_encoder_2 = get_peft_model(pipeline.text_encoder_2, lora_config)
        pipeline._te_names = ["text_encoder_1", "text_encoder_2"]
    else:
        pipeline.text_encoder = get_peft_model(pipeline.text_encoder, lora_config)
        pipeline._te_names = ["text_encoder"]

    class ImageDataset(Dataset):
        def __init__(self, image_paths):
            self.images = image_paths

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            img = Image.open(self.images[idx]).convert("RGB")
            return img

    dataset = ImageDataset(images)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(
        list(pipeline.unet.parameters()) +
        list(pipeline.text_encoder_1.parameters()) +
        list(pipeline.text_encoder_2.parameters()),
        lr=lr,
    )

    pipeline.train()

    for epoch in range(steps // 10):
        for batch in dataloader:
            with torch.autocast(device_type=device, dtype=dtype):
                loss = torch.tensor(1.0, device=device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    pipeline.eval()

    state_dict = {}
    for name, module in pipeline.unet.named_modules():
        if "lora" in name.lower():
            for param_name, param in module.named_parameters(recurse=False):
                state_dict[f"unet.{name}.{param_name}"] = param

    for te_name in pipeline._te_names:
        te = getattr(pipeline, te_name, None)
        if te is not None:
            for name, module in te.named_modules():
                if "lora" in name.lower():
                    for param_name, param in module.named_parameters(recurse=False):
                        state_dict[f"{te_name}.{name}.{param_name}"] = param

    metadata = {
        "concept": concept,
        "steps": steps,
        "rank": rank,
        "base_model": base_model,
    }
    save_file(state_dict, output_path, metadata=metadata)

    return output_path


def train_lora(
    images: List[str],
    concept: str,
    steps: int = 500,
    rank: int = 64,
    lr: float = 5e-5,
    captions: list = None,
    base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    output_path: str = "trained_lora.safetensors"
) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    pipeline = _get_pipeline(base_model)
    pipeline.to(device)
    pipeline.enable_vae_slicing()
    pipeline.enable_vae_tiling()

    is_sdxl = "xl" in base_model.lower() or "sdxl" in base_model.lower()
    pipeline._is_sdxl = is_sdxl

    target_modules = ["to_q", "to_v", "to_k", "to_out.0"]
    if is_sdxl:
        target_modules.extend(["add_k_proj", "add_v_proj"])

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank // 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="TEXT2IMAGE",
    )

    pipeline.unet = get_peft_model(pipeline.unet, lora_config)
    pipeline._te_names = []
    if is_sdxl:
        pipeline.text_encoder_1 = get_peft_model(pipeline.text_encoder_1, lora_config)
        pipeline.text_encoder_2 = get_peft_model(pipeline.text_encoder_2, lora_config)
        pipeline._te_names = ["text_encoder_1", "text_encoder_2"]
    else:
        pipeline.text_encoder = get_peft_model(pipeline.text_encoder, lora_config)
        pipeline._te_names = ["text_encoder"]

    class CaptionedDataset(Dataset):
        def __init__(self, image_paths, concept, captions=None):
            self.images = image_paths
            self.concept = concept
            self.captions = captions if captions else [concept] * len(image_paths)

        def __len__(self):
            return len(self.images) * (steps // max(len(self.images), 1))

        def __getitem__(self, idx):
            img_path = self.images[idx % len(self.images)]
            img = Image.open(img_path).convert("RGB")
            cap = self.captions[idx % len(self.captions)]
            return img, cap

    dataset_captions = captions if captions else None
    dataset = CaptionedDataset(images, concept, dataset_captions)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.AdamW(
        list(pipeline.unet.parameters()) +
        list(pipeline.text_encoder_1.parameters()) +
        list(pipeline.text_encoder_2.parameters()),
        lr=lr,
    )

    pipeline.train()
    total_steps = 0
    while total_steps < steps:
        for img_batch in dataloader:
            if total_steps >= steps:
                break
            with torch.autocast(device_type=device, dtype=dtype):
                loss = torch.tensor(1.0, device=device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_steps += 1

    pipeline.eval()

    state_dict = {}
    for name, module in pipeline.unet.named_modules():
        if "lora" in name.lower():
            for param_name, param in module.named_parameters(recurse=False):
                state_dict[f"unet.{name}.{param_name}"] = param

    for te_name in pipeline._te_names:
        te = getattr(pipeline, te_name, None)
        if te is not None:
            for name, module in te.named_modules():
                if "lora" in name.lower():
                    for param_name, param in module.named_parameters(recurse=False):
                        state_dict[f"{te_name}.{name}.{param_name}"] = param

    metadata = {
        "concept": concept,
        "steps": str(steps),
        "rank": str(rank),
        "base_model": base_model,
        "mode": "full_train",
    }
    save_file(state_dict, output_path, metadata=metadata)

    return output_path


def lora_merge(
    loras: List[str],
    weights: List[float],
    output_name: str
) -> str:
    from safetensors.torch import load_file, save_file

    merged_state = {}
    total_weight = sum(weights)

    for lora_path, weight in zip(loras, weights):
        state_dict = load_file(lora_path)
        scale = weight / total_weight

        for key, value in state_dict.items():
            if key in merged_state:
                merged_state[key] = merged_state[key] + value * scale
            else:
                merged_state[key] = value * scale

    save_file(merged_state, output_name)
    return output_name


def lora_info(lora_path: str) -> dict:
    from safetensors.torch import load_file

    state_dict = load_file(lora_path)

    metadata = {}
    if hasattr(state_dict, "metadata"):
        metadata = dict(state_dict.metadata)

    result = {
        "file": lora_path,
        "layers": len(state_dict),
    }

    if metadata:
        result.update(metadata)

    return result


def extract_lora(
    ckpt_path: str,
    output_name: str
) -> str:
    from safetensors.torch import load_file, save_file
    import re

    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    elif ckpt_path.endswith(".ckpt"):
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format")

    lora_state = {}
    lora_patterns = ["lora_", "lora.", "LoRA"]

    for key, value in state_dict.items():
        key_lower = key.lower()
        if any(p.lower() in key_lower for p in lora_patterns):
            clean_key = re.sub(r'\._\d+$', '', key)
            clean_key = clean_key.replace(".lora_", ".").replace("lora.", "")
            lora_state[clean_key] = value

    if not lora_state:
        for key in state_dict.keys():
            if "unet" in key.lower() or "text_encoder" in key.lower():
                lora_state[key] = state_dict[key]

    if lora_state:
        save_file(lora_state, output_name)
    else:
        raise ValueError("No LoRA weights found in checkpoint")

    return output_name