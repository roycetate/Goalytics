# Goalytics

Goalytics is a football video analytics project that uses computer vision to detect players, referees, and the ball, then turns those detections into live match metrics. It combines a Python detection agent, a FastAPI backend, and a React dashboard for viewing live stats, analysis frames, heatmaps, and processed videos.

## Features

- YOLOv8-based player and ball detection.
- ByteTrack/Ultralytics tracking for stable player identities.
- Two-team classification from jersey color clustering.
- Referee filtering with HSV/LAB color heuristics.
- Ball smoothing and recovery with a Kalman filter.
- Live possession, touches, player counts, FPS, and tracking quality metrics.
- Optional pitch homography for radar points, shot, shot-on-target, and goal heuristics.
- Live annotated frame stream consumed by the dashboard.
- Team heatmap generation at the end of analysis.
- React dashboard pages for live match overview, insights, stream preview, and processed videos.

## Project Structure

```text
GOALYTICS-PROJECT/
|-- agent/
|   `-- agent.py              # Computer vision analysis pipeline
|-- backend/
|   `-- server.py             # FastAPI API for stats, frames, videos, and heatmaps
|-- data/
|   |-- live_stats.json       # Runtime stats written by the agent
|   `-- live_frame.jpg        # Latest annotated frame written by the agent
|-- frontend/
|   |-- src/
|   |   |-- api.js
|   |   |-- App.jsx
|   |   `-- pages/
|   |-- package.json
|   `-- vite.config.js
|-- yolov8m.pt                # YOLO model weights
`-- PROJECT_OVERVIEW.md       # High-level architecture and workflow notes
```

## Requirements

### Python

- Python 3.10 or newer recommended.
- `fastapi`
- `uvicorn`
- `opencv-python`
- `numpy`
- `torch`
- `ultralytics`
- `easyocr` optional, only needed when running the agent with `--enable-ocr`.

### Frontend

- Node.js 20 or newer recommended.
- npm.

## Setup

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install optional OCR support:

```bash
pip install easyocr
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

## Running the Project

Run the backend API from the project root:

```bash
uvicorn backend.server:app --reload --port 8765
```

Run the frontend in a second terminal:

```bash
cd frontend
npm run dev
```

Open the Vite URL shown in the terminal, usually:

```text
http://localhost:5173
```

Run the analysis agent in a third terminal:

```bash
python agent/agent.py --video agent/RMvsLVP2.mp4 --model yolov8m.pt --no-show
```

The agent writes live outputs into `data/`, which the backend serves to the frontend.

You can also upload a match from the dashboard instead of copying a file into `agent/`:

1. Open [http://localhost:5173/videos](http://localhost:5173/videos)
2. Drop an MP4 (or MOV/MKV/WEBM/AVI) onto the upload panel
3. Click **Analyze selected**
4. Watch live stats on `/dashboard` while the agent runs

Uploaded files are stored in `uploads/`. The backend starts `agent/agent.py` against that file and writes the usual live frame, stats, heatmap, and `agent/output_detection.mp4` outputs. Use **Stop** on the Videos page to cancel a running job.

## Useful Agent Options

```bash
python agent/agent.py \
  --video agent/RMvsLVP2.mp4 \
  --model yolov8m.pt \
  --out agent/output_detection.mp4 \
  --device auto \
  --no-show
```

Common options:

- `--video`: input match video.
- `--model`: YOLO model path.
- `--out`: processed output video path.
- `--device`: `auto`, `cpu`, `mps`, `0`, `cuda:0`, etc.
- `--pitch-polygon`: normalized pitch boundary points for filtering off-pitch detections.
- `--homography-src` and `--homography-dst`: normalized point pairs for mapping video coordinates to a 2D pitch.
- `--enable-ocr`: enables jersey number OCR if EasyOCR is installed.
- `--stats-out`: JSON stats output path, defaults to `data/live_stats.json`.
- `--live-frame-out`: annotated frame output path, defaults to `data/live_frame.jpg`.

## Backend API

Default frontend configuration expects the API at `http://127.0.0.1:8765` when running Vite on port `5173`.

Available routes:

- `GET /`: backend health message.
- `GET /stats`: latest JSON stats from `data/live_stats.json`.
- `GET /analysis-frame`: latest annotated JPEG frame.
- `GET /analysis-stream`: MJPEG stream of the latest annotated frames.
- `GET /heatmaps`: available generated heatmaps.
- `GET /heatmaps/{team}`: heatmap image for `team1` or `team2`.
- `GET /videos`: processed video files found in `agent/`.
- `GET /videos/{video_name}`: serves a processed video file.

## Frontend Pages

- `/dashboard`: live match dashboard, scoreboard, possession, tracking metrics, and stream preview.
- `/insights`: team-specific possession, activity, and heatmap insights.
- `/analysis-stream`: full live analysis stream.
- `/videos`: processed video browser for files in `agent/`.

## Runtime Outputs

The analysis agent produces:

- `data/live_stats.json`: live match metrics consumed by `/stats`.
- `data/live_frame.jpg`: latest annotated frame consumed by `/analysis-frame` and `/analysis-stream`.
- `agent/output_detection.mp4`: default processed output video.
- `agent/team1_pitch.png`: Team 1 heatmap generated after analysis completes.
- `agent/team2_pitch.png`: Team 2 heatmap generated after analysis completes.

## Development Notes

- If the frontend cannot reach the backend, set `VITE_API_BASE` before starting Vite:

```bash
VITE_API_BASE=http://127.0.0.1:8765 npm run dev
```

- CPU inference can be slow. Use `--device mps` on Apple Silicon or a CUDA device when available.
- For cleaner detections, provide `--pitch-polygon` so the agent can ignore crowd, staff, and sideline objects outside the playing area.
- Shot and goal heuristics require homography data so ball movement can be evaluated in pitch coordinates.
