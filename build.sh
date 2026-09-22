#!/usr/bin/env bash
set -euo pipefail

# Railway/BuildKit can reuse cached base layers. Fail early if the native
# toolchain is missing rather than letting InsightFace fail deep inside pip.
if ! command -v g++ >/dev/null 2>&1; then
  echo "[DEEPCAM BUILD] g++ missing; installing native build toolchain"
  apt-get update
  apt-get install -y --no-install-recommends gcc g++ make build-essential
  rm -rf /var/lib/apt/lists/*
fi
g++ --version

python -m pip install --upgrade pip
python -m pip install -r backend/requirements.txt

# Use Deep-Live-Cam's own model downloader so URLs, expected sizes,
# resume support, and the InsightFace buffalo_l pack remain consistent
# with the upstream engine.
python - <<'PY'
from modules.model_downloader import ensure_model, ensure_insightface_pack

required = [
    "inswapper_128_fp16.onnx",
    "gfpgan-1024.onnx",
]

for name in required:
    path = ensure_model(name)
    if not path:
        raise SystemExit(f"Failed to obtain required model: {name}")
    print(f"[DEEPCAM BUILD] ready: {path}")

if not ensure_insightface_pack("buffalo_l"):
    raise SystemExit("Failed to obtain required InsightFace buffalo_l model pack")

print("[DEEPCAM BUILD] model assets ready")
PY
