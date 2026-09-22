import asyncio
import os

import httpx
import websockets
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

UPSTREAM = os.getenv("DEEPCAM_INFERENCE_URL", "").rstrip("/")
if not UPSTREAM:
    raise RuntimeError("DEEPCAM_INFERENCE_URL is required for the Deepcam gateway")

UPSTREAM_WS = UPSTREAM.replace("https://", "wss://", 1).replace("http://", "ws://", 1)

app = FastAPI(title="Deepcam Gateway", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTTP_TIMEOUT = httpx.Timeout(
    connect=15.0,
    read=float(os.getenv("DEEPCAM_GATEWAY_READ_TIMEOUT", "180")),
    write=30.0,
    pool=15.0,
)

@app.get("/")
async def root():
    return {
        "service": "deepcam-gateway",
        "status": "ok",
        "health": "/health",
        "info": "/info",
        "process": "/process",
        "live_source": "/live/source",
        "live_ws": "/live/ws/{session_id}",
    }

@app.get("/health")
async def health():
    # Gateway health remains cheap, while upstream readiness is checked by /info.
    return {"status": "ok", "service": "deepcam-gateway"}

@app.get("/info")
async def info():
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            r = await client.get(f"{UPSTREAM}/info")
            return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Inference backend unavailable: {exc}")

@app.post("/stream-swap")
async def stream_swap(
    source_image: UploadFile = File(...),
    target_frame_stream: UploadFile = File(...),
):
    source_raw = await source_image.read()
    target_raw = await target_frame_stream.read()
    if not source_raw or not target_raw:
        raise HTTPException(status_code=400, detail="Source and target frame are required")
    files = {
        "source_image": (source_image.filename or "source.jpg", source_raw, source_image.content_type or "image/jpeg"),
        "target_frame_stream": (target_frame_stream.filename or "frame.jpg", target_raw, target_frame_stream.content_type or "image/jpeg"),
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            r = await client.post(f"{UPSTREAM}/stream-swap", files=files)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Inference backend unavailable: {exc}")
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/octet-stream"))

@app.post("/live/source")
async def live_source(source: UploadFile = File(...)):
    raw = await source.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Source photo is empty")
    files = {
        "source": (source.filename or "source.jpg", raw, source.content_type or "image/jpeg"),
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            r = await client.post(f"{UPSTREAM}/live/source", files=files)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Inference backend unavailable: {exc}")
    if r.status_code >= 400:
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
    data = r.json()
    session_id = data.get("session_id")
    if not session_id:
        raise HTTPException(status_code=502, detail="Inference backend returned no live session")
    return {
        "session_id": session_id,
        "websocket": f"/live/ws/{session_id}",
        "status": data.get("status", "ready"),
    }

@app.websocket("/live/ws/{session_id}")
async def live_ws(websocket: WebSocket, session_id: str):
    await websocket.accept()
    upstream_url = f"{UPSTREAM_WS}/live/ws/{session_id}"
    try:
        async with websockets.connect(
            upstream_url,
            max_size=4 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=5,
        ) as upstream:

            async def client_to_upstream():
                while True:
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("bytes") is not None:
                        await upstream.send(msg["bytes"])
                    elif msg.get("text") is not None:
                        await upstream.send(msg["text"])

            async def upstream_to_client():
                while True:
                    msg = await upstream.recv()
                    if isinstance(msg, bytes):
                        await websocket.send_bytes(msg)
                    else:
                        await websocket.send_text(msg)

            tasks = [
                asyncio.create_task(client_to_upstream()),
                asyncio.create_task(upstream_to_client()),
            ]
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                exc = task.exception()
                if exc and not isinstance(exc, (WebSocketDisconnect, websockets.exceptions.ConnectionClosed)):
                    raise exc
    except (WebSocketDisconnect, websockets.exceptions.ConnectionClosed):
        pass
    except Exception as exc:
        try:
            await websocket.close(code=1011, reason=f"Gateway upstream error: {exc}")
        except Exception:
            pass

@app.post("/process")
async def process(
    source: UploadFile = File(...),
    target: UploadFile = File(...),
):
    source_raw = await source.read()
    target_raw = await target.read()
    if not source_raw or not target_raw:
        raise HTTPException(status_code=400, detail="Source and target are required")
    files = {
        "source": (source.filename or "source.jpg", source_raw, source.content_type or "application/octet-stream"),
        "target": (target.filename or "target.mp4", target_raw, target.content_type or "application/octet-stream"),
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=900.0, write=60.0, pool=15.0)) as client:
        try:
            r = await client.post(f"{UPSTREAM}/process", files=files)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Inference backend unavailable: {exc}")
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/octet-stream"))
