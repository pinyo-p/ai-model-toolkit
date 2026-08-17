FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7800

CMD ["python", "main.py"]
