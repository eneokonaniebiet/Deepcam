import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests


ROOT = Path(__file__).resolve().parent
REQUIRED = ["app.py", "Dockerfile", "build.sh", "requirements.txt", "render.yaml"]


def verify_local_codebase_integrity():
    print("Running local architectural checks...")
    missing = [name for name in REQUIRED if not (ROOT / name).exists()]
    if missing:
        raise SystemExit(f"Missing deployment files: {', '.join(missing)}")

    print("  OK: deployment files present")

    dummy = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(dummy, (100, 100), 45, (255, 255, 255), -1)
    cv2.imwrite(str(ROOT / "dummy_source.jpg"), dummy)
    cv2.imwrite(str(ROOT / "dummy_target.jpg"), dummy)
    print("  OK: validation images created")


def execute_remote_cluster_test(endpoint_base_url):
    base = endpoint_base_url.rstrip("/")
    print(f"Testing: {base}")

    response = requests.get(f"{base}/health", timeout=20)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") not in {"ok", "healthy"}:
        raise SystemExit(f"Unexpected health payload: {payload}")
    print("  OK: /health")

    with open(ROOT / "dummy_source.jpg", "rb") as src, open(ROOT / "dummy_target.jpg", "rb") as tgt:
        files = {
            "source_image": ("src.jpg", src, "image/jpeg"),
            "target_frame_stream": ("tgt.jpg", tgt, "image/jpeg"),
        }
        result = requests.post(f"{base}/stream-swap", files=files, timeout=60)

    if result.status_code != 200:
        raise SystemExit(f"/stream-swap failed: {result.status_code}: {result.text[:500]}")

    content_type = result.headers.get("content-type", "").split(";")[0].lower()
    if content_type != "image/jpeg":
        raise SystemExit(f"Unexpected content type: {content_type}")

    proof = ROOT / "deployment_proof.jpg"
    proof.write_bytes(result.content)
    decoded = cv2.imdecode(np.frombuffer(result.content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise SystemExit("Returned JPEG could not be decoded")

    print(f"  OK: /stream-swap returned {len(result.content)} bytes")
    print(f"  OK: proof written to {proof}")


if __name__ == "__main__":
    verify_local_codebase_integrity()
    if len(sys.argv) > 1:
        execute_remote_cluster_test(sys.argv[1])
    else:
        print("Local checks complete. Pass the deployed base URL to run the remote inference check.")
