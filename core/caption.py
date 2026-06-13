import torch
import json
import re
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import BlipProcessor, BlipForConditionalGeneration


_model = None
_processor = None


def _get_model():
    global _model, _processor
    if _model is None:
        dtype = torch.float16
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=dtype
        )
        if device == "cuda":
            _model = _model.to(device)
        _model.eval()
    return _model, _processor


def image_captioning(image_path: str) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = _get_model()

    image = Image.open(image_path).convert("RGB")

    inputs = processor(image, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=100)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def _read_exif_caption(image_path: str) -> str | None:
    try:
        img = Image.open(image_path)
        exif_data = img._getexif()
        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, tag_id)
                if tag_name in ("ImageDescription", "XPComment", "UserComment"):
                    if isinstance(value, bytes):
                        try:
                            value = value.decode("utf-8", errors="ignore").strip("\x00").strip()
                        except:
                            continue
                    if value and isinstance(value, str) and len(value) > 5:
                        return value

        info = img.info
        if "parameters" in info:
            params = info["parameters"]
            first_line = params.split("\n")[0].strip()
            if first_line and len(first_line) > 5:
                return first_line
        if "Description" in info:
            desc = info["Description"]
            if isinstance(desc, str) and len(desc) > 5:
                return desc
        if "comment" in info:
            comment = info["comment"]
            if isinstance(comment, bytes):
                comment = comment.decode("utf-8", errors="ignore")
            if isinstance(comment, str) and len(comment) > 5:
                return comment

        return None
    except Exception:
        return None


def auto_caption(image_paths: list[str]) -> list[dict]:
    results = []
    need_blip = []

    for path in image_paths:
        caption = _read_exif_caption(path)
        if caption:
            results.append({"path": path, "source": "metadata", "caption": caption})
        else:
            need_blip.append(path)

    if need_blip:
        try:
            model, processor = _get_model()
            device = "cuda" if torch.cuda.is_available() else "cpu"

            for path in need_blip:
                image = Image.open(path).convert("RGB")
                inputs = processor(image, return_tensors="pt")
                if device == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = model.generate(**inputs, max_new_tokens=100)
                caption = processor.decode(output[0], skip_special_tokens=True)
                results.append({"path": path, "source": "blip", "caption": caption})
        except Exception:
            for path in need_blip:
                results.append({"path": path, "source": "blip", "caption": ""})

    return results