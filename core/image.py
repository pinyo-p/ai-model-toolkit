from PIL import Image
import numpy as np


_model = None


def _get_model():
    global _model
    if _model is None:
        try:
            from realesrgan_ncnn_vulkan import RealESRGANer
        except:
            pass
        try:
            import cv2
            from basicsr.archs.rrdb_arch import RRDBNet
            from basicsr.models.realesrgan_model import RealESRGANModel
        except:
            pass
        _model = "cv2"
    return _model


def upscale(image: Image.Image, scale: int = 4) -> Image.Image:
    try:
        import cv2
        import numpy as np

        img_array = np.array(image)

        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)

        new_width = img_array.shape[1] * scale
        new_height = img_array.shape[0] * scale

        upscaled = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        upscaled_pil = Image.fromarray(cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB))

        return upscaled_pil
    except Exception as e:
        width, height = image.size
        new_size = (width * scale, height * scale)
        return image.resize(new_size, Image.Resampling.LANCZOS)