import subprocess
import re
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
        result["vram_free_gb"] = round((torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3, 2)
        result["cuda_version"] = torch.version.cuda or ""
        return result

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if out.returncode != 0 or not out.stdout.strip():
            return result

        line = out.stdout.strip().split("\n")[0].strip()
        parts = [p.strip() for p in line.split(",")]

        result["cuda_available"] = True
        result["gpu_name"] = parts[0] if parts else "Unknown GPU"

        if len(parts) > 1 and parts[1] and parts[1] not in ("N/A", "[Not Supported]", ""):
            try:
                result["vram_total_gb"] = round(float(parts[1]) / 1024, 2)
            except ValueError:
                pass
        if len(parts) > 2 and parts[2] and parts[2] not in ("N/A", "[Not Supported]", ""):
            try:
                result["vram_free_gb"] = round(float(parts[2]) / 1024, 2)
            except ValueError:
                pass

        ver_out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        m = re.search(r"CUDA Version:\s*([\d.]+)", ver_out.stdout)
        if m:
            result["cuda_version"] = m.group(1)
    except Exception:
        pass

    return result
