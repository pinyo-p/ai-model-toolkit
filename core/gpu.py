import torch


def check_gpu() -> dict:
    result = {
        "cuda_available": False,
        "gpu_name": "CPU",
        "vram_total_gb": 0,
        "vram_free_gb": 0,
        "cuda_version": None,
    }

    if torch.cuda.is_available():
        result["cuda_available"] = True
        result["gpu_name"] = torch.cuda.get_device_name(0)
        result["vram_total_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        result["vram_free_gb"] = round(torch.cuda.memory_allocated(0) / 1024**3, 2)
        result["cuda_version"] = torch.version.cuda
    else:
        result["cuda_available"] = False
        result["gpu_name"] = "No GPU"

    return result