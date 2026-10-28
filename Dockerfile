FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 libsndfile1 fontconfig fonts-noto-core && rm -rf /var/lib/apt/lists/*

# CPU torch by default; for an NVIDIA GPU build pass
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV HF_HOME=/models/hf
# Usage: docker run ... video-clipper --url <video url>
ENTRYPOINT ["python", "run_pipeline.py"]
