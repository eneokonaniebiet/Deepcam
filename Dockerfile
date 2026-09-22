FROM python:3.12-slim

# Cache-bust native toolchain layer after compiler-related fixes.
ARG DEEPCAM_BUILD_TOOLCHAIN_REV=20260922-2

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
    libglib2.0-0 \
    curl \
    git \
    ca-certificates \
    && gcc --version && g++ --version && make --version \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace

RUN chmod +x /workspace/build.sh && /workspace/build.sh

EXPOSE 10000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "10000"]
