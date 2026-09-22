import asyncio
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import numpy as np

APP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.getenv("DEEPCAM_WORK_DIR", "/tmp/deepcam"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)

import static_ffmpeg
static_ffmpeg.add_paths(weak=True)

app = FastAPI(title="Deepcam Backend", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])
PROCESS_LOCK = threading.Lock()
LIVE_LOCK = threading.Lock()
LIVE_SESSIONS = {}
LIVE_SESSION_TTL_SECONDS = 30 * 60
LIVE_MAX_FRAME_BYTES = int(os.getenv("DEEPCAM_LIVE_MAX_FRAME_BYTES", str(2 * 1024 * 1024)))
LIVE_JPEG_QUALITY = int(os.getenv("DEEPCAM_LIVE_JPEG_QUALITY", "82"))
LIVE_MAX_WIDTH = int(os.getenv("DEEPCAM_LIVE_MAX_WIDTH", "640"))
LIVE_MAX_HEIGHT = int(os.getenv("DEEPCAM_LIVE_MAX_HEIGHT", "480"))
_RUNTIME_READY = False

@app.get("/")
def root():
    return {"service":"deepcam-api","status":"ok","health":"/health","info":"/info","process":"/process","live_source":"/live/source","live_ws":"/live/ws/{session_id}"}

@app.get("/health")
def health():
    return {"status":"ok","service":"deepcam-api"}

@app.get("/info")
def info():
    import onnxruntime
    return {"providers":onnxruntime.get_available_providers(),"configured_provider":os.getenv("DEEPCAM_EXECUTION_PROVIDER","cpu"),"live":True}

async def save_upload(upload: UploadFile, path: Path) -> None:
    with path.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk: break
            f.write(chunk)

def run_deepcam(source_path: Path, target_path: Path, output_path: Path) -> None:
    sys.path.insert(0, str(APP_ROOT))
    import modules.globals
    from modules import core
    modules.globals.source_path=str(source_path)
    modules.globals.target_path=str(target_path)
    modules.globals.output_path=str(output_path)
    modules.globals.frame_processors=["face_swapper"]
    modules.globals.headless=True
    modules.globals.keep_fps=True
    modules.globals.keep_audio=True
    modules.globals.keep_frames=False
    modules.globals.many_faces=False
    modules.globals.nsfw_filter=True
    modules.globals.map_faces=False
    modules.globals.mouth_mask=False
    modules.globals.video_encoder="libx264"
    modules.globals.video_quality=18
    modules.globals.live_mirror=False
    modules.globals.live_resizable=False
    modules.globals.max_memory=int(os.getenv("DEEPCAM_MAX_MEMORY_GB","1"))
    modules.globals.execution_threads=int(os.getenv("DEEPCAM_EXECUTION_THREADS","2"))
    modules.globals.execution_providers=core.decode_execution_providers(os.getenv("DEEPCAM_EXECUTION_PROVIDER","cpu").split(","))
    if not modules.globals.execution_providers:
        modules.globals.execution_providers=core.decode_execution_providers(["cpu"])
    if not core.pre_check(): raise RuntimeError("Deepcam runtime pre-check failed")
    for processor in core.get_frame_processors_modules(modules.globals.frame_processors):
        if not processor.pre_check(): raise RuntimeError(f"Frame processor pre-check failed: {processor.NAME}")
    core.limit_resources()
    core.start()
    if not output_path.exists(): raise RuntimeError("Deepcam did not produce an output file")

def configure_live_runtime() -> None:
    global _RUNTIME_READY
    if _RUNTIME_READY:
        return
    sys.path.insert(0, str(APP_ROOT))
    import modules.globals
    from modules import core
    modules.globals.frame_processors = ["face_swapper"]
    modules.globals.headless = True
    modules.globals.many_faces = False
    modules.globals.map_faces = False
    modules.globals.mouth_mask = False
    modules.globals.nsfw_filter = False
    modules.globals.opacity = 1.0
    modules.globals.sharpness = 0.0
    modules.globals.enable_interpolation = False
    modules.globals.execution_threads = int(os.getenv("DEEPCAM_EXECUTION_THREADS","2"))
    modules.globals.max_memory = int(os.getenv("DEEPCAM_MAX_MEMORY_GB","1"))
    modules.globals.execution_providers = core.decode_execution_providers(
        os.getenv("DEEPCAM_EXECUTION_PROVIDER","cpu").split(",")
    )
    if not modules.globals.execution_providers:
        modules.globals.execution_providers = core.decode_execution_providers(["cpu"])
    if not core.pre_check():
        raise RuntimeError("Deepcam runtime pre-check failed")
    for processor in core.get_frame_processors_modules(["face_swapper"]):
        if not processor.pre_check():
            raise RuntimeError(f"Frame processor pre-check failed: {processor.NAME}")
    core.limit_resources()
    from modules.face_analyser import get_face_analyser
    from modules.processors.frame.face_swapper import get_face_swapper
    get_face_analyser()
    if get_face_swapper() is None:
        raise RuntimeError("Face swapper model could not be loaded")
    _RUNTIME_READY = True


def prepare_live_source(raw: bytes) -> str:
    configure_live_runtime()
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The uploaded source is not a readable image")
    from modules.face_analyser import get_one_face
    source_face = get_one_face(image)
    if source_face is None:
        raise ValueError("No face was detected in the uploaded source photo")
    cutoff = time.time() - LIVE_SESSION_TTL_SECONDS
    for sid in list(LIVE_SESSIONS):
        if LIVE_SESSIONS[sid]["created_at"] < cutoff:
            LIVE_SESSIONS.pop(sid, None)
    sid = uuid.uuid4().hex
    LIVE_SESSIONS[sid] = {"source_face": source_face, "created_at": time.time()}
    return sid


def process_live_frame(source_face, raw: bytes) -> bytes:
    if len(raw) > LIVE_MAX_FRAME_BYTES:
        raise ValueError("Camera frame is too large")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid camera frame")
    h, w = frame.shape[:2]
    if w > LIVE_MAX_WIDTH or h > LIVE_MAX_HEIGHT:
        scale = min(LIVE_MAX_WIDTH / w, LIVE_MAX_HEIGHT / h)
        frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    from modules.face_analyser import detect_one_face_fast
    from modules.processors.frame.face_swapper import swap_face
    target_face = detect_one_face_fast(frame)
    if target_face is not None:
        frame = swap_face(source_face, target_face, frame)
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, max(50, min(95, LIVE_JPEG_QUALITY))])
    if not ok:
        raise RuntimeError("Failed to encode processed camera frame")
    return encoded.tobytes()


@app.post("/live/source")
async def live_source(source: UploadFile = File(...)):
    raw = await source.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Source photo is empty")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Source photo is too large")
    try:
        sid = await asyncio.to_thread(prepare_live_source, raw)
        return {"session_id": sid, "websocket": f"/live/ws/{sid}", "status": "ready"}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.websocket("/live/ws/{session_id}")
async def live_ws(websocket: WebSocket, session_id: str):
    session = LIVE_SESSIONS.get(session_id)
    if session is None:
        await websocket.close(code=1008, reason="Live session not found or expired")
        return
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            raw = message.get("bytes")
            if raw is None:
                text = message.get("text")
                if text and text.lower() in {"ping", "keepalive"}:
                    await websocket.send_text("pong")
                continue
            try:
                with LIVE_LOCK:
                    result = await asyncio.to_thread(process_live_frame, session["source_face"], raw)
                await websocket.send_bytes(result)
                session["created_at"] = time.time()
            except ValueError as exc:
                await websocket.send_json({"error": str(exc)})
            except Exception as exc:
                await websocket.send_json({"error": f"Live frame processing failed: {exc}"})
    except WebSocketDisconnect:
        pass
    finally:
        LIVE_SESSIONS.pop(session_id, None)


@app.post("/process")
async def process(source: UploadFile=File(...), target: UploadFile=File(...)):
    source_name=Path(source.filename or "source.jpg").name
    target_name=Path(target.filename or "target.mp4").name
    target_ext=Path(target_name).suffix.lower() or ".mp4"
    job_dir=Path(tempfile.mkdtemp(prefix="job-",dir=WORK_ROOT))
    source_path=job_dir/source_name
    target_path=job_dir/target_name
    output_path=job_dir/f"deepcam-output{target_ext}"
    try:
        await save_upload(source,source_path)
        await save_upload(target,target_path)
        with PROCESS_LOCK: run_deepcam(source_path,target_path,output_path)
        media_type="video/mp4" if target_ext in {".mp4",".m4v",".mov"} else "application/octet-stream"
        return FileResponse(output_path,media_type=media_type,filename=output_path.name)
    except Exception as exc:
        raise HTTPException(status_code=500,detail=str(exc))
