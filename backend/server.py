import json
import re
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.jobs import (
    UPLOAD_DIR,
    load_match_config,
    manager,
    save_match_config,
)

app = FastAPI(title="Goalytics API")

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "live_stats.json"
LIVE_FRAME_FILE = ROOT / "data" / "live_frame.jpg"
VIDEO_DIR = ROOT / "agent"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
HEATMAP_FILES = {
    "team1": ROOT / "agent" / "team1_pitch.png",
    "team2": ROOT / "agent" / "team2_pitch.png",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MatchConfigUpdate(BaseModel):
    team1_name: str | None = None
    team2_name: str | None = None
    team1_color: str | None = None
    team2_color: str | None = None


class AnalysisStartRequest(BaseModel):
    video_name: str = Field(..., min_length=1)
    team1_name: str | None = None
    team2_name: str | None = None
    team1_color: str | None = None
    team2_color: str | None = None


def safe_video_name(name: str) -> str:
    base = Path(name).name
    suffix = Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(base).stem).strip("._")[:80]
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    if not stem:
        stem = "match"
    return f"{stem}{suffix}"


def unique_upload_path(filename: str) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    candidate = UPLOAD_DIR / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    return UPLOAD_DIR / f"{stem}_{int(time.time())}{suffix}"


def video_entry(path: Path, kind: str) -> dict:
    return {
        "name": path.name,
        "kind": kind,
        "size_bytes": path.stat().st_size,
        "url": f"/videos/{path.name}",
        "updated_at": path.stat().st_mtime,
    }


def list_dir_videos(directory: Path, kind: str) -> list[dict]:
    if not directory.exists():
        return []
    files = []
    for item in directory.iterdir():
        if item.is_file() and item.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            if kind == "processed" and not item.name.startswith("output"):
                continue
            if kind == "library" and item.name.startswith("output"):
                continue
            files.append(video_entry(item, kind))
    files.sort(key=lambda item: item["name"].lower())
    return files


def resolve_video_file(video_name: str) -> Path | None:
    name = Path(video_name).name
    for directory in (UPLOAD_DIR, VIDEO_DIR):
        candidate = (directory / name).resolve()
        if not str(candidate).startswith(str(directory.resolve())):
            continue
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_stats_payload() -> dict:
    payload: dict = {"available": False, "message": "No data available yet"}
    if DATA_FILE.exists():
        try:
            payload = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            payload["available"] = True
        except (OSError, json.JSONDecodeError):
            payload = {"available": False, "message": "Live stats file is unreadable"}
    payload["match"] = load_match_config()
    payload["analysis"] = manager.snapshot()
    return payload


@app.get("/")
def home():
    return {"message": "Goalytics Backend Running ⚽"}


@app.get("/stats")
def get_stats():
    return load_stats_payload()


@app.get("/match-config")
def get_match_config():
    return load_match_config()


@app.put("/match-config")
def update_match_config(update: MatchConfigUpdate):
    return save_match_config(update.model_dump(exclude_none=True))


@app.get("/analysis")
def get_analysis():
    return manager.snapshot()


@app.post("/analysis/start")
def start_analysis(request: AnalysisStartRequest):
    video_path = resolve_video_file(request.video_name)
    if video_path is None:
        raise HTTPException(status_code=404, detail="Video not found")
    save_match_config(request.model_dump(exclude_none=True))
    try:
        return manager.start(video_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/analysis/stop")
def stop_analysis():
    try:
        return manager.stop()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/uploads")
async def upload_video(file: UploadFile = File(...)):
    filename = safe_video_name(file.filename or "match.mp4")
    dest = unique_upload_path(filename)
    written = 0
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="File exceeds 500 MB limit")
                handle.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file was empty")

    return video_entry(dest, "upload")


@app.get("/analysis-frame")
def get_analysis_frame():
    if not LIVE_FRAME_FILE.exists():
        raise HTTPException(status_code=404, detail="No live analysis frame available yet")

    return FileResponse(
        path=LIVE_FRAME_FILE,
        media_type="image/jpeg",
        filename=LIVE_FRAME_FILE.name,
        headers={"Cache-Control": "no-store"},
    )


def live_frame_stream():
    last_mtime = 0.0

    while True:
        if LIVE_FRAME_FILE.exists():
            mtime = LIVE_FRAME_FILE.stat().st_mtime
            if mtime != last_mtime:
                last_mtime = mtime
                frame = LIVE_FRAME_FILE.read_bytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Cache-Control: no-store\r\n"
                    b"\r\n"
                    + frame
                    + b"\r\n"
                )
        time.sleep(0.03)


@app.get("/analysis-stream")
def get_analysis_stream():
    return StreamingResponse(
        live_frame_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/heatmaps")
def list_heatmaps():
    heatmaps = []
    for team, path in HEATMAP_FILES.items():
        if path.exists():
            stat = path.stat()
            heatmaps.append(
                {
                    "team": team,
                    "label": "Team 1" if team == "team1" else "Team 2",
                    "url": f"/heatmaps/{team}",
                    "updated_at": stat.st_mtime,
                    "size_bytes": stat.st_size,
                }
            )
    return {"heatmaps": heatmaps}


@app.get("/heatmaps/{team}")
def get_heatmap(team: str):
    path = HEATMAP_FILES.get(team)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown heatmap")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Heatmap not available yet")

    return FileResponse(
        path=path,
        media_type="image/png",
        filename=path.name,
        headers={"Cache-Control": "no-store"},
    )


@app.get("/videos")
def list_videos():
    uploads = list_dir_videos(UPLOAD_DIR, "upload")
    library = list_dir_videos(VIDEO_DIR, "library")
    processed = list_dir_videos(VIDEO_DIR, "processed")
    return {
        "uploads": uploads,
        "library": library,
        "processed": processed,
        "videos": uploads + library + processed,
    }


@app.get("/videos/{video_name}")
def get_video(video_name: str):
    candidate = resolve_video_file(video_name)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Video not found")
    if candidate.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return FileResponse(path=candidate, filename=candidate.name)
