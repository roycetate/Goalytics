import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

app = FastAPI(title="Goalytics API")

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "live_stats.json"
LIVE_FRAME_FILE = ROOT / "data" / "live_frame.jpg"
VIDEO_DIR = ROOT / "agent"
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
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


@app.get("/")
def home():
    return {"message": "Goalytics Backend Running ⚽"}


@app.get("/stats")
def get_stats():
    if DATA_FILE.exists():
        with DATA_FILE.open(encoding="utf-8") as f:
            return json.load(f)
    return {"message": "No data available yet"}


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
                    b"Cache-Control: no-store\r\n\r\n"
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
    if not VIDEO_DIR.exists():
        return {"videos": []}

    files = []
    for item in VIDEO_DIR.iterdir():
        if item.is_file() and item.suffix.lower() in ALLOWED_VIDEO_EXTENSIONS:
            files.append(
                {
                    "name": item.name,
                    "size_bytes": item.stat().st_size,
                    "url": f"/videos/{item.name}",
                }
            )
    files.sort(key=lambda x: x["name"])
    return {"videos": files}


@app.get("/videos/{video_name}")
def get_video(video_name: str):
    candidate = (VIDEO_DIR / video_name).resolve()

    if not str(candidate).startswith(str(VIDEO_DIR.resolve())):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    if candidate.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    return FileResponse(path=candidate, filename=candidate.name)
