FROM python:3.12-slim

# Cache-bust native toolchain layer after compiler/runtime dependency fixes.
ARG DEEPCAM_BUILD_TOOLCHAIN_REV=20260922-3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN echo "[DEEPCAM BUILD] toolchain rev ${DEEPCAM_BUILD_TOOLCHAIN_REV}" \
    && apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    build-essential \
    ffmpeg \
    libgl1 \
    libegl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    git \
    ca-certificates \
    && gcc --version && g++ --version && make --version \
    && ldconfig -p | grep -E 'libEGL.so.1|libGL.so.1' \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN chmod +x /workspace/build.sh && /workspace/build.sh

EXPOSE 10000
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "10000"]
