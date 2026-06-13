FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir fastapi==0.115.0 uvicorn==0.30.6 diffusers==0.30.3 transformers==4.44.2 accelerate==0.33.0 safetensors==0.4.4 peft==0.12.0 bitsandbytes==0.43.3 Pillow==10.4.0 python-multipart==0.0.9 realesrgan==0.3.0 salesforce-lavis==1.0.2 lycoris-lora==2.1.0

COPY . .

EXPOSE 7800

CMD ["python", "main.py"]