import os
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

APP_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.getenv("DEEPCAM_WORK_DIR", "/tmp/deepcam"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)

import static_ffmpeg
static_ffmpeg.add_paths(weak=True)

app = FastAPI(title="Deepcam Backend", version="1.0.0")
PROCESS_LOCK = threading.Lock()

@app.get("/")
def root():
    return {"service":"deepcam-api","status":"ok","health":"/health","info":"/info","process":"/process"}

@app.get("/health")
def health():
    return {"status":"ok","service":"deepcam-api"}

@app.get("/info")
def info():
    import onnxruntime
    return {"providers":onnxruntime.get_available_providers(),"configured_provider":os.getenv("DEEPCAM_EXECUTION_PROVIDER","cpu")}

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
