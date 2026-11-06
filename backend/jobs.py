import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT_SCRIPT = ROOT / "agent" / "agent.py"
AGENT_DIR = ROOT / "agent"
UPLOAD_DIR = ROOT / "uploads"
DATA_DIR = ROOT / "data"
STATUS_FILE = DATA_DIR / "analysis_status.json"
LOG_FILE = DATA_DIR / "analysis.log"
MATCH_CONFIG_FILE = DATA_DIR / "match_config.json"
DEFAULT_OUTPUT = AGENT_DIR / "output_detection.mp4"

DEFAULT_MATCH_CONFIG = {
    "team1_name": "Liverpool",
    "team2_name": "Real Madrid",
    "team1_color": "#EF4444",
    "team2_color": "#FFFFFF",
}

MODEL_CANDIDATES = (
    ROOT / "yolov8m.pt",
    AGENT_DIR / "yolov8m.pt",
    AGENT_DIR / "yolov8m.mlpackage",
)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_match_config() -> dict:
    if not MATCH_CONFIG_FILE.exists():
        return dict(DEFAULT_MATCH_CONFIG)
    try:
        data = json.loads(MATCH_CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_MATCH_CONFIG)
    merged = dict(DEFAULT_MATCH_CONFIG)
    for key in DEFAULT_MATCH_CONFIG:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            merged[key] = value.strip()
    return merged


def save_match_config(updates: dict) -> dict:
    current = load_match_config()
    for key in DEFAULT_MATCH_CONFIG:
        value = updates.get(key)
        if isinstance(value, str) and value.strip():
            current[key] = value.strip()
    write_json_atomic(MATCH_CONFIG_FILE, current)
    return current


def resolve_model_path() -> Path | None:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    return None


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class AnalysisManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.process: subprocess.Popen | None = None
        self.state = self._load_state()
        self._reconcile_running()

    def snapshot(self) -> dict:
        with self.lock:
            self._reconcile_running_locked()
            return dict(self.state)

    def start(self, video_path: Path) -> dict:
        model = resolve_model_path()
        if model is None:
            raise FileNotFoundError(
                "No YOLO model found. Place yolov8m.pt in the project root or agent/ folder."
            )
        if not AGENT_SCRIPT.exists():
            raise FileNotFoundError("agent/agent.py is missing.")

        with self.lock:
            self._reconcile_running_locked()
            if self.state.get("status") == "running":
                raise RuntimeError("Analysis is already running.")

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            log_handle = LOG_FILE.open("ab", buffering=0)
            cmd = [
                sys.executable,
                str(AGENT_SCRIPT),
                "--video",
                str(video_path),
                "--model",
                str(model),
                "--out",
                str(DEFAULT_OUTPUT),
                "--device",
                "auto",
                "--no-show",
            ]
            self.process = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.state = {
                "status": "running",
                "video_name": video_path.name,
                "video_path": str(video_path),
                "model": str(model),
                "output_video": str(DEFAULT_OUTPUT.relative_to(ROOT)),
                "pid": self.process.pid,
                "started_at": time.time(),
                "finished_at": None,
                "return_code": None,
                "error": None,
            }
            self._persist_locked()
            threading.Thread(target=self._watch, daemon=True).start()
            return dict(self.state)

    def stop(self) -> dict:
        with self.lock:
            self._reconcile_running_locked()
            pid = self.state.get("pid")
            if self.state.get("status") != "running" or not pid:
                raise RuntimeError("No analysis job is running.")
            try:
                os.killpg(pid, signal.SIGTERM)
            except OSError:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError as exc:
                    raise RuntimeError(f"Could not stop analysis: {exc}") from exc
            self.state["status"] = "stopping"
            self.state["error"] = None
            self._persist_locked()
            return dict(self.state)

    def _watch(self) -> None:
        process = self.process
        if process is None:
            return
        return_code = process.wait()
        with self.lock:
            if self.process is process:
                self.process = None
            if self.state.get("pid") != process.pid:
                return
            self.state["return_code"] = return_code
            self.state["finished_at"] = time.time()
            if self.state.get("status") == "stopping":
                self.state["status"] = "stopped"
            elif return_code == 0:
                self.state["status"] = "completed"
                self.state["error"] = None
            else:
                self.state["status"] = "failed"
                self.state["error"] = (
                    f"Agent exited with code {return_code}. See data/analysis.log."
                )
            self._persist_locked()

    def _reconcile_running(self) -> None:
        with self.lock:
            self._reconcile_running_locked()

    def _reconcile_running_locked(self) -> None:
        if self.state.get("status") != "running":
            return
        pid = self.state.get("pid")
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                return
            self.state["return_code"] = code
            self.state["finished_at"] = self.state.get("finished_at") or time.time()
            self.state["status"] = "completed" if code == 0 else "failed"
            if code != 0:
                self.state["error"] = f"Agent exited with code {code}. See data/analysis.log."
            self.process = None
            self._persist_locked()
            return
        if pid_alive(pid):
            return
        self.state["status"] = "failed"
        self.state["finished_at"] = time.time()
        self.state["error"] = "Analysis process is no longer running."
        self._persist_locked()

    def _load_state(self) -> dict:
        default = {
            "status": "idle",
            "video_name": None,
            "video_path": None,
            "model": None,
            "output_video": None,
            "pid": None,
            "started_at": None,
            "finished_at": None,
            "return_code": None,
            "error": None,
        }
        if not STATUS_FILE.exists():
            return default
        try:
            data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default
        default.update(data)
        return default

    def _persist_locked(self) -> None:
        write_json_atomic(STATUS_FILE, self.state)


manager = AnalysisManager()
