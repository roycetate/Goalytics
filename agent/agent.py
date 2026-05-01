import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO



# COCO indices for the default YOLOv8 pretrained weights
CLASS_PERSON = 0
CLASS_SPORTS_BALL = 32

PREVIEW_WIN = "Goalytics Detection"

# BGR draw colors
COLOR_TEAM1 = (0, 255, 0)  # green
COLOR_TEAM2 = (255, 96, 0)  # orange-ish (BGR)
COLOR_REF = (255, 0, 255)  # magenta
COLOR_BALL = (0, 0, 255)  # red

class BallKalmanFilter:
    """Simple 4-state Kalman filter: [x, y, vx, vy]"""
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix = np.array([
            [1,0,1,0],
            [0,1,0,1],
            [0,0,1,0],
            [0,0,0,1]
        ], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def update(self, cx: float, cy: float):
        measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
        if not self.initialized:
            self.kf.statePre = np.array([[cx],[cy],[0],[0]], np.float32)
            self.kf.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
            self.initialized = True
        self.kf.correct(measurement)
        self.kf.predict()

    def predict(self) -> tuple[float, float] | None:
        if not self.initialized:
            return None
        pred = self.kf.predict()
        return float(pred[0][0]), float(pred[1][0])

def preview_window_ok() -> bool:
    try:
        cv2.namedWindow(PREVIEW_WIN, cv2.WINDOW_NORMAL)
        return True
    except cv2.error:
        return False


def parse_args():
    p = argparse.ArgumentParser(
        description="Goalytics: pitch-only detection, two teams (jersey colors), referee heuristic."
    )
    here = Path(__file__).resolve().parent
    p.add_argument("--video", type=Path, default=here / "RMmatch.mov")
    p.add_argument("--model", type=Path, default=here / "yolov8m.pt")
    p.add_argument(
        "--out",
        type=Path,
        default=here / "output_detection.mp4",
        help="Output video path",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference size (long side). Larger helps tiny ball / distant players; slower.",
    )
    p.add_argument(
        "--conf-min",
        type=float,
        default=0.06,
        help="Low threshold for NMS; raise per-class filters below so the ball is not discarded early.",
    )
    p.add_argument(
        "--conf-person",
        type=float,
        default=0.28,
        help="Min confidence to keep a person box after inference.",
    )
    p.add_argument(
        "--conf-ball",
        type=float,
        default=0.05,
        help="Min confidence to keep sports ball (usually needs to be lower than person).",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS IoU. Slightly lower can improve recall when many overlapping players.",
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="Test-time augmentation; slower, sometimes helps hard frames.",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a preview window (faster, good for servers).",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Inference device: auto (CUDA if available, else Apple MPS, else CPU), "
            "or cpu, mps, 0, cuda:0, etc."
        ),
    )
    p.add_argument(
        "--pitch-polygon",
        type=str,
        default="",
        help=(
            "8 numbers: x1,y1,x2,y2,x3,y3,x4,y4 in normalized 0–1 image coords (clockwise). "
            "When set, persons and the ball are kept only if inside the pitch. Required for "
            "strict off-pitch suppression. Example: 0.02,0.14,0.98,0.14,0.98,0.99,0.02,0.99"
        ),
    )
    p.add_argument(
        "--pitch-lift-top",
        type=float,
        default=0.0,
        help="Move the polygon's top edge upward by this many normalized units.",
    )
    p.add_argument(
        "--pitch-on",
        type=str,
        choices=("foot", "lenient"),
        default="foot",
        help="On-pitch test for people: foot center, or lenient (bottom corners too).",
    )
    p.add_argument(
        "--person-edge-margin",
        type=float,
        default=12.0,
        help=(
            "Minimum distance in pixels from pitch polygon edge for person/ref foot points. "
            "Helps remove cameraman/staff standing on or just outside touchlines."
        ),
    )
    p.add_argument(
        "--draw-pitch-roi",
        action="store_true",
        help="Draw the pitch polygon in yellow (useful while tuning).",
    )
    p.add_argument(
        "--referee-tag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use HSV heuristic for teal/light-blue kits as referee (default: on).",
    )
    p.add_argument(
        "--hide-referee",
        action="store_true",
        help="Do not draw referee boxes (still excludes them from team clustering).",
    )
    p.add_argument(
        "--teams",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Split non-referee players into two teams by jersey LAB color (default: on).",
    )
    p.add_argument(
        "--referee-hsv-low",
        type=str,
        default="88,45,90",
        help="Fallback HSV box if teal-fraction is low (H,S,V OpenCV).",
    )
    p.add_argument(
        "--referee-hsv-high",
        type=str,
        default="118,255,255",
        help="Fallback HSV upper bound.",
    )
    p.add_argument(
        "--referee-max-lab-a",
        type=float,
        default=133.0,
        help=(
            "OpenCV LAB 'a' channel (0–255, ~128 neutral). Red kits are usually higher; "
            "above this => never referee. Lowers false 'referee' on red players."
        ),
    )
    p.add_argument(
        "--referee-min-teal-frac",
        type=float,
        default=0.22,
        help=(
            "Among torso pixels with decent S/V, fraction whose hue lies in the cyan/teal band "
            "must exceed this to count as referee (stops red/white being called ref)."
        ),
    )
    p.add_argument(
        "--referee-teal-hue",
        type=str,
        default="85,118",
        help="Comma-separated Hmin,Hmax (OpenCV hue 0–179) for teal/cyan referee shirts.",
    )
    p.add_argument(
        "--track",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use tracker IDs (ByteTrack) for stable player/ball identities (default: on).",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config (e.g., bytetrack.yaml or botsort.yaml).",
    )
    p.add_argument(
        "--stats-out",
        type=Path,
        default=here.parent / "data" / "live_stats.json",
        help="JSON file for live stats consumed by backend /stats.",
    )
    p.add_argument(
        "--stats-every",
        type=int,
        default=5,
        help="Write stats every N frames (default 5).",
    )
    p.add_argument(
        "--possession-radius",
        type=float,
        default=120.0,
        help="Max pixel distance from ball center to player foot-center to assign possession.",
    )
    p.add_argument(
        "--possession-switch-frames",
        type=int,
        default=3,
        help="Require this many consecutive frames before switching ball owner/team.",
    )
    p.add_argument(
        "--require-grass-foot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Count person/ref only if a patch around feet looks like grass (default: on).",
    )
    p.add_argument(
        "--grass-hue",
        type=str,
        default="32,98",
        help="Green hue band Hmin,Hmax (OpenCV HSV hue 0-179) used by --require-grass-foot.",
    )
    p.add_argument(
        "--grass-min-frac",
        type=float,
        default=0.18,
        help="Minimum fraction of green pixels in the foot patch to keep a person as on-pitch.",
    )
    p.add_argument(
        "--grass-below-min-frac",
        type=float,
        default=0.16,
        help="Minimum green fraction in a patch just below the feet (strong sideline-staff filter).",
    )
    p.add_argument(
        "--grass-below-offset",
        type=float,
        default=6.0,
        help="Pixel offset below feet center for the grass-below test.",
    )
    p.add_argument(
        "--ball-recover",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a second low-conf ball-only pass around previous ball when full-frame misses.",
    )
    p.add_argument(
        "--ball-recover-conf",
        type=float,
        default=0.03,
        help="Confidence threshold for ball recovery pass.",
    )
    p.add_argument(
        "--ball-recover-scale",
        type=float,
        default=3.0,
        help="Expand previous ball box by this scale to create recovery ROI.",
    )
    p.add_argument(
        "--ball-recover-max-miss",
        type=int,
        default=15,
        help="How many consecutive missed frames to keep trying ROI ball recovery.",
    )
    return p.parse_args()


def parse_hsv_triplet(s: str) -> np.ndarray:
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated ints, got: {s!r}")
    return np.array(parts, dtype=np.float32)


def parse_hue_pair(s: str) -> tuple[float, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected 2 comma-separated numbers, got: {s!r}")
    return parts[0], parts[1]


def is_patch_grass(
    frame: np.ndarray,
    x: float,
    y: float,
    hue_min: float,
    hue_max: float,
    min_frac: float,
) -> bool:
    h, w = frame.shape[:2]
    cx = int(round(x))
    cy = int(round(y))
    # Small local patch near the player's foot point.
    r = max(6, int(0.012 * min(w, h)))
    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)
    if x2 <= x1 or y2 <= y1:
        return False
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return False
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    # Exclude very low saturation/value (line paint, shadows, dark clothing).
    good = hsv[(hsv[:, 1] >= 28) & (hsv[:, 2] >= 35)]
    if good.shape[0] < 12:
        return False
    hch = good[:, 0].astype(np.float32)
    green = (hch >= hue_min) & (hch <= hue_max)
    return float(np.mean(green)) >= float(min_frac)


def is_grass_below_feet(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    hue_min: float,
    hue_max: float,
    min_frac: float,
    offset_px: float,
) -> bool:
    cx = 0.5 * (x1 + x2)
    bh = max(1.0, float(y2 - y1))
    y_probe = float(y2) + max(float(offset_px), 0.05 * bh)
    return is_patch_grass(frame, cx, y_probe, hue_min, hue_max, min_frac)


def expand_bbox(x1: int, y1: int, x2: int, y2: int, scale: float, w: int, h: int) -> tuple[int, int, int, int]:
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(2.0, float(x2 - x1) * max(1.2, scale))
    bh = max(2.0, float(y2 - y1) * max(1.2, scale))
    rx1 = max(0, int(round(cx - bw * 0.5)))
    ry1 = max(0, int(round(cy - bh * 0.5)))
    rx2 = min(w, int(round(cx + bw * 0.5)))
    ry2 = min(h, int(round(cy + bh * 0.5)))
    return rx1, ry1, rx2, ry2


def parse_norm_pitch_polygon(s: str) -> np.ndarray | None:
    s = s.strip()
    if not s:
        return None
    parts = [float(x.strip()) for x in s.replace(";", ",").split(",")]
    if len(parts) != 8:
        raise ValueError(
            "--pitch-polygon needs exactly 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4 (normalized 0–1)"
        )
    pts = np.clip(np.array(parts, dtype=np.float32).reshape(4, 2), 0.0, 1.0)
    return pts


def polygon_to_pixels(poly_norm: np.ndarray, w: int, h: int) -> np.ndarray:
    p = poly_norm.copy()
    p[:, 0] *= w
    p[:, 1] *= h
    return np.round(p).astype(np.int32)


def lift_polygon_top_edge(poly_norm: np.ndarray, lift: float) -> np.ndarray:
    if lift <= 0:
        return poly_norm
    p = poly_norm.copy()
    min_y = float(p[:, 1].min())
    tol = 1e-4
    for i in range(4):
        if p[i, 1] <= min_y + tol:
            p[i, 1] = max(0.0, float(p[i, 1]) - lift)
    return p


def foot_on_pitch(x1: int, y1: int, x2: int, y2: int, poly_px: np.ndarray) -> bool:
    cx = (x1 + x2) * 0.5
    y_foot = float(y2)
    return cv2.pointPolygonTest(poly_px, (cx, y_foot), False) >= 0


def person_on_pitch(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    poly_px: np.ndarray,
    mode: str,
    min_edge_margin: float = 0.0,
) -> bool:
    def inside_with_margin(px: float, py: float) -> bool:
        # Signed distance to polygon edge: >0 inside, <0 outside.
        dist = cv2.pointPolygonTest(poly_px, (px, py), True)
        return dist >= float(min_edge_margin)

    if mode == "foot":
        return inside_with_margin((x1 + x2) * 0.5, float(y2))
    y_foot = float(y2)
    xs = ((x1 + x2) * 0.5, float(x1), float(x2))
    for x in xs:
        if inside_with_margin(x, y_foot):
            return True
    return False


def ball_on_pitch(x1: int, y1: int, x2: int, y2: int, poly_px: np.ndarray) -> bool:
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    return cv2.pointPolygonTest(poly_px, (cx, cy), False) >= 0


def torso_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 4:
        return None
    mx = int(0.22 * bw)
    my_top = int(0.12 * bh)
    my_bot = int(0.52 * bh)
    xa, xb = max(0, x1 + mx), min(w, x2 - mx)
    ya, yb = max(0, y1 + my_top), min(h, y1 + my_bot)
    if xb <= xa or yb <= ya:
        return None
    return frame[ya:yb, xa:xb]


def mean_hsv(bgr: np.ndarray) -> np.ndarray | None:
    if bgr.size == 0:
        return None
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0).astype(np.float32)


def mean_lab(bgr: np.ndarray) -> np.ndarray | None:
    if bgr.size == 0:
        return None
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)


def teal_hue_fraction(
    bgr: np.ndarray,
    h_min: float,
    h_max: float,
    sat_min: int = 42,
    val_min: int = 38,
) -> float:
    """Share of saturated torso pixels whose hue is in the cyan/teal band (red kits are ~0–15)."""
    if bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    sel = hsv[(hsv[:, 1] >= sat_min) & (hsv[:, 2] >= val_min)]
    if sel.shape[0] < 8:
        return 0.0
    h = sel[:, 0].astype(np.float32)
    in_band = (h >= h_min) & (h <= h_max)
    return float(np.mean(in_band))


def is_referee_jersey(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    hsv_low: np.ndarray,
    hsv_high: np.ndarray,
    max_lab_a: float,
    min_teal_frac: float,
    teal_h_min: float,
    teal_h_max: float,
) -> bool:
    """
    Referee = cyan/teal shirt, not red/white tricked by mean HSV.
    Red kits push LAB 'a' above neutral; we require teal hue among saturated pixels.
    """
    crop = torso_crop(frame, x1, y1, x2, y2)
    if crop is None:
        return False

    lab = mean_lab(crop)
    if lab is not None and float(lab[1]) > max_lab_a:
        return False

    frac = teal_hue_fraction(crop, teal_h_min, teal_h_max)
    if frac >= min_teal_frac:
        return True

    # Pale broadcast blue: fewer saturated pixels than deep teal, but LAB 'a' sits below neutral
    # (white kits stay near a~128; red already excluded by max_lab_a).
    if lab is not None:
        L, la = float(lab[0]), float(lab[1])
        weak_need = max(0.09, min_teal_frac * 0.42)
        if la <= max_lab_a and 112.0 <= la <= 124.0 and L >= 112.0 and frac >= weak_need:
            return True

    m = mean_hsv(crop)
    if m is None:
        return False
    if float(m[1]) < float(hsv_low[1]) or float(m[2]) < float(hsv_low[2]):
        return False
    return bool(np.all(m >= hsv_low) and np.all(m <= hsv_high))


def kmeans_two_lab(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """feats (n,3) LAB. Returns labels (n,), centroids (2,3)."""
    n = feats.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int32), np.zeros((2, 3), dtype=np.float32)
    if n == 1:
        v = feats[0]
        return np.zeros(1, dtype=np.int32), np.stack([v, v])
    i_dark = int(np.argmin(feats[:, 0]))
    i_light = int(np.argmax(feats[:, 0]))
    if i_light == i_dark:
        i_light = (i_light + 1) % n
    c0 = feats[i_dark].copy()
    c1 = feats[i_light].copy()
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(30):
        d0 = np.sum((feats - c0) ** 2, axis=1)
        d1 = np.sum((feats - c1) ** 2, axis=1)
        new_labels = (d1 < d0).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        m0 = feats[labels == 0]
        m1 = feats[labels == 1]
        if m0.size:
            c0 = m0.mean(axis=0)
        if m1.size:
            c1 = m1.mean(axis=0)
    if not np.any(labels == 0) or not np.any(labels == 1):
        c0 = feats.min(axis=0)
        c1 = feats.max(axis=0)
        d0 = np.sum((feats - c0) ** 2, axis=1)
        d1 = np.sum((feats - c1) ** 2, axis=1)
        labels = (d1 < d0).astype(np.int32)
    centroids = np.stack([c0, c1])
    return labels, centroids


def align_team_ids(
    prev_centroids: np.ndarray | None, centroids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map k-means cluster id -> stable team 0/1. Returns (aligned_centroids (2,3), remap[2])
    where stable_team = remap[kmeans_label].
    """
    if prev_centroids is None:
        order = np.argsort(centroids[:, 0])
        aligned = centroids[order]
        remap = np.empty(2, dtype=np.int32)
        remap[order[0]] = 0
        remap[order[1]] = 1
        return aligned, remap
    d_same = np.linalg.norm(centroids[0] - prev_centroids[0]) + np.linalg.norm(
        centroids[1] - prev_centroids[1]
    )
    d_swap = np.linalg.norm(centroids[0] - prev_centroids[1]) + np.linalg.norm(
        centroids[1] - prev_centroids[0]
    )
    if d_swap < d_same:
        aligned = np.stack([centroids[1], centroids[0]])
        remap = np.array([1, 0], dtype=np.int32)
    else:
        aligned = centroids.copy()
        remap = np.array([0, 1], dtype=np.int32)
    return aligned, remap


def min_conf_for_class(cls_id: int, conf_person: float, conf_ball: float) -> float:
    if cls_id == CLASS_SPORTS_BALL:
        return conf_ball
    return conf_person


def resolve_device(name: str) -> str:
    n = (name or "auto").strip().lower()
    if n == "auto":
        if torch.cuda.is_available():
            return "0"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    return name.strip()


def want_half_precision(device: str) -> bool:
    if not torch.cuda.is_available():
        return False
    d = device.lower()
    if d in ("cpu", "mps"):
        return False
    return d.startswith("cuda") or d.isdigit()



def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    tmp.replace(path)

def generate_pitch_heatmap(positions, pitch, output_path, frame_w, frame_h):
    import numpy as np
    import cv2

    h, w = pitch.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    # Scale positions to pitch size
    for x, y in positions:
        px = int(x * w / frame_w)
        py = int(y * h / frame_h)

        if 0 <= px < w and 0 <= py < h:
            heatmap[py, px] += 1

    # Smooth
    heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)

    # Normalize
    heatmap = heatmap / (heatmap.max() + 1e-6)
    heatmap = (heatmap * 255).astype(np.uint8)

    # Color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay on pitch
    overlay = cv2.addWeighted(pitch, 0.6, heatmap_color, 0.4, 0)

    cv2.imwrite(output_path, overlay)

def generate_heatmap_plot(positions, width, height, output_path):
    import numpy as np
    import cv2

    heatmap = np.zeros((height, width), dtype=np.float32)

    for x, y in positions:
        if 0 <= x < width and 0 <= y < height:
            heatmap[int(y), int(x)] += 1

    # 🔥 Smooth it (ONLY ONCE)
    heatmap = cv2.GaussianBlur(heatmap, (41, 41), 0)

    # 🔥 Better normalization (ONLY THIS)
    max_val = heatmap.max()
    heatmap = heatmap / (max_val + 1e-6)

    heatmap = np.power(heatmap, 0.5)

    # Convert to 0–255
    heatmap = (heatmap * 255).astype(np.uint8)

    # Color map
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Save image
    cv2.imwrite(output_path, heatmap_color)

def generate_heatmap(frame, positions):
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    for (x, y) in positions:
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            heatmap[y, x] += 1

    heatmap = cv2.GaussianBlur(heatmap, (25, 25), 0)
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heatmap_color, 0.4, 0)

    return overlay


def main():
    args = parse_args()

    pitch = cv2.imread("pitch.jpg")

    if pitch is None:
        print("Error: pitch.jpg not found")
        return

    # Resize for better quality
    pitch = cv2.resize(pitch, (1050, 680))

    video_path = args.video
    if not video_path.is_file():
        print(f"Video not found: {video_path}")
        return

    pitch_norm = parse_norm_pitch_polygon(args.pitch_polygon)
    ref_low = parse_hsv_triplet(args.referee_hsv_low)
    ref_high = parse_hsv_triplet(args.referee_hsv_high)
    teal_h_min, teal_h_max = parse_hue_pair(args.referee_teal_hue)
    grass_h_min, grass_h_max = parse_hue_pair(args.grass_hue)

    device = resolve_device(args.device)
    use_half = want_half_precision(device)
    model = YOLO(str(args.model))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error opening video")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = int(round(fps)) if fps >= 1 else 25

    out = cv2.VideoWriter(
        str(args.out),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )

    print(
        f"Inference: device={device}, half={use_half}, imgsz={args.imgsz}, model={args.model.name}"
    )
    if pitch_norm is None:
        print(
            "Warning: no --pitch-polygon — crowd/staff off the field will still be detected. "
            "Set the polygon to restrict everything to the pitch."
        )
    else:
        print(
            "Pitch mask: ON — persons (foot rule) and ball (center) must lie inside the polygon."
        )
    print(
        f"Referee heuristic: {'on' if args.referee_tag else 'off'}; "
        f"two teams: {'on' if args.teams else 'off'}."
    )

    if device == "cpu":
        print(
            "Tip: CPU is slow. On Apple Silicon use --device mps; lower --imgsz or yolov8n.pt for speed."
        )

    show_preview = not args.no_show
    if args.no_show:
        print("Live preview: OFF (--no-show). Output is still written to --out.")
    elif not preview_window_ok():
        print(
            "Live preview: disabled — OpenCV could not create a window. "
            "Use `pip install opencv-python` (not headless)."
        )
        show_preview = False
    else:
        print(
            "Live preview: ON. First frame may take a few seconds (model warmup)."
        )

    print(
        f"Tracking: {'on' if args.track else 'off'} ({args.tracker}); "
        f"live stats -> {args.stats_out} every {max(1, args.stats_every)} frame(s)."
    )

    prev_team_centroids: np.ndarray | None = None
    frame_idx = 0
    t_prev = time.perf_counter()
    live_fps = 0.0
    stats_every = max(1, int(args.stats_every))
    team_pos_seconds = {0: 0.0, 1: 0.0}
    team_touch_counts = {0: 0, 1: 0}
    player_touch_counts: dict[int, int] = {}
    player_positions = {}
    player_distance = {}
    previous_owner_track_id = None
    previous_owner_team = None
    passes = 0
    current_owner_team: int | None = None
    current_owner_track_id: int | None = None
    pending_owner_team: int | None = None
    pending_owner_track_id: int | None = None
    pending_owner_frames = 0
    switch_frames = max(1, int(args.possession_switch_frames))
    last_ball_box: tuple[int, int, int, int] | None = None
    ball_miss_frames = 0
    ball_kalman = BallKalmanFilter()

    frame_h = frame_height
    frame_w = frame_width

    # 🔥 TEAM HEATMAP STORAGE (GLOBAL FOR WHOLE VIDEO)
    team1_positions = []
    team2_positions = []

    while True:
        frame_idx += 1
        ret, frame = cap.read()
        if not ret:
            break

        poly_px = None
        if pitch_norm is not None:
            poly_use = lift_polygon_top_edge(pitch_norm, args.pitch_lift_top)
            poly_px = polygon_to_pixels(poly_use, frame_width, frame_height)
            if args.draw_pitch_roi:
                cv2.polylines(
                    frame,
                    [poly_px],
                    isClosed=True,
                    color=(0, 255, 255),
                    thickness=2,
                )

        infer_kwargs = dict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf_min,
            iou=args.iou,
            classes=[CLASS_PERSON, CLASS_SPORTS_BALL],
            augment=args.augment,
            max_det=300,
            verbose=False,
            half=use_half,
            device=device,
        )
        if args.track:
            results = model.track(
                **infer_kwargs,
                persist=True,
                tracker=args.tracker,
            )
        else:
            results = model.predict(**infer_kwargs)

        raw: list[tuple] = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < min_conf_for_class(cls, args.conf_person, args.conf_ball):
                    continue
                label = model.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = None
                if getattr(box, "id", None) is not None:
                    track_id = int(box.id[0])
                raw.append((label, conf, x1, y1, x2, y2, track_id))

        # Ball recovery pass: if full-frame misses, look around last known ball location.
        saw_ball_in_raw = any(item[0] == "sports ball" for item in raw)
        if (
            args.ball_recover
            and (not saw_ball_in_raw)
            and last_ball_box is not None
            and ball_miss_frames <= max(0, int(args.ball_recover_max_miss))
        ):
            rx1, ry1, rx2, ry2 = expand_bbox(
                last_ball_box[0],
                last_ball_box[1],
                last_ball_box[2],
                last_ball_box[3],
                float(args.ball_recover_scale),
                frame_width,
                frame_height,
            )
            if rx2 > rx1 and ry2 > ry1:
                roi = frame[ry1:ry2, rx1:rx2]
                rec = model.predict(
                    source=roi,
                    imgsz=max(640, int(args.imgsz * 0.7)),
                    conf=float(args.ball_recover_conf),
                    iou=args.iou,
                    classes=[CLASS_SPORTS_BALL],
                    max_det=10,
                    verbose=False,
                    half=use_half,
                    device=device,
                )
                for rr in rec:
                    for box in rr.boxes:
                        conf = float(box.conf[0])
                        if conf < args.conf_ball:
                            continue
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        raw.append(
                            (
                                "sports ball",
                                conf,
                                bx1 + rx1,
                                by1 + ry1,
                                bx2 + rx1,
                                by2 + ry1,
                                None,
                            )
                        )

        players: list[tuple[int, int, int, int, float, bool, np.ndarray | None, int | None]] = []
        balls: list[tuple[int, int, int, int, float, int | None]] = []

        for label, conf, x1, y1, x2, y2, track_id in raw:
            if label == "person":
                if poly_px is not None and not person_on_pitch(
                    x1, y1, x2, y2, poly_px, args.pitch_on, args.person_edge_margin
                ):
                    continue
                if args.require_grass_foot:
                    foot_x = 0.5 * (x1 + x2)
                    foot_y = float(y2)
                    foot_grass_ok = is_patch_grass(
                        frame,
                        foot_x,
                        foot_y,
                        grass_h_min,
                        grass_h_max,
                        args.grass_min_frac,
                    )
                    below_grass_ok = is_grass_below_feet(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        grass_h_min,
                        grass_h_max,
                        args.grass_below_min_frac,
                        args.grass_below_offset,
                    )
                    if not (foot_grass_ok and below_grass_ok):
                        continue
                is_ref = False
                if args.referee_tag:
                    is_ref = is_referee_jersey(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        ref_low,
                        ref_high,
                        args.referee_max_lab_a,
                        args.referee_min_teal_frac,
                        teal_h_min,
                        teal_h_max,
                    )
                lab = None
                if args.teams and not is_ref:
                    crop = torso_crop(frame, x1, y1, x2, y2)
                    if crop is not None:
                        lab = mean_lab(crop)
                players.append((x1, y1, x2, y2, conf, is_ref, lab, track_id))
            elif label == "sports ball":
                if poly_px is not None and not ball_on_pitch(x1, y1, x2, y2, poly_px):
                    continue
                balls.append((x1, y1, x2, y2, conf, track_id))

        if balls:
            best_ball = max(balls, key=lambda b: b[4])
            last_ball_box = (best_ball[0], best_ball[1], best_ball[2], best_ball[3])
            ball_miss_frames = 0
            # ✅ Update Kalman with real detection
            bcx = 0.5 * (best_ball[0] + best_ball[2])
            bcy = 0.5 * (best_ball[1] + best_ball[3])
            ball_kalman.update(bcx, bcy)
        else:
            ball_miss_frames += 1
            # ✅ Use Kalman prediction as synthetic ball position for possession/stats
            pred = ball_kalman.predict()
            if pred is not None and ball_miss_frames <= args.ball_recover_max_miss:
                px, py = pred
                r = 12  # synthetic radius
                synthetic_ball = (
                    int(px - r), int(py - r),
                    int(px + r), int(py + r),
                    0.0,  # conf=0 means "predicted, not detected"
                    None
                )
                balls = [synthetic_ball]
                # Draw predicted ball differently
                cv2.circle(frame, (int(px), int(py)), r, (0, 165, 255), 2)  # orange = predicted
                cv2.putText(frame, "ball~pred", (int(px) - 10, int(py) - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        team_remap = np.array([0, 1], dtype=np.int32)
        if args.teams and len(players) >= 2:
            idx_non_ref = [i for i, p in enumerate(players) if not p[5] and p[6] is not None]
            if len(idx_non_ref) >= 2:
                feats = np.stack([players[i][6] for i in idx_non_ref])
                km_labels, centroids = kmeans_two_lab(feats)
                aligned, team_remap = align_team_ids(prev_team_centroids, centroids)
                prev_team_centroids = aligned
                for row, pi in enumerate(idx_non_ref):
                    lst = list(players[pi])
                    k = int(km_labels[row])
                    lst.append(int(team_remap[k]))
                    players[pi] = tuple(lst)
            else:
                for i, p in enumerate(players):
                    if not p[5] and p[6] is not None:
                        pl = list(p)
                        pl.append(0)
                        players[i] = tuple(pl)
        else:
            for i, p in enumerate(players):
                if not p[5]:
                    pl = list(p)
                    pl.append(0)
                    players[i] = tuple(pl)

        for p in players:
            x1, y1, x2, y2 = p[0], p[1], p[2], p[3]
            conf = p[4]
            is_ref = p[5]
            track_id = p[7]

            # =========================
            # 📍 TRACK PLAYER POSITIONS
            # =========================
            if track_id is not None and not is_ref:
                px = int((x1 + x2) / 2)
                py = int(y2)  # foot position

                if track_id not in player_positions:
                    player_positions[track_id] = []

                player_positions[track_id].append((px, py))

                if track_id not in player_distance:
                    player_distance[track_id] = 0.0

                if len(player_positions[track_id]) > 1:
                    prev_x, prev_y = player_positions[track_id][-2]
                    curr_x, curr_y = player_positions[track_id][-1]

                    player_distance[track_id] += np.hypot(
                        curr_x - prev_x,
                        curr_y - prev_y
                    )

                # limit memory (keep last 100 points)
                if len(player_positions[track_id]) > 100:
                    player_positions[track_id] = player_positions[track_id][-100:]

            # =========================
            # 🎨 DRAWING LOGIC
            # =========================
            if is_ref:
                if args.hide_referee:
                    continue
                display_label = "referee"
                color = COLOR_REF

            elif args.teams and len(p) >= 9:
                tid = p[8]
                display_label = f"team{tid + 1}"
                color = COLOR_TEAM1 if tid == 0 else COLOR_TEAM2

            else:
                display_label = "player"
                color = COLOR_TEAM1

            # 🔥 DRAW MOVEMENT TRAILS

            if track_id is not None and track_id in player_positions:
                pts = player_positions[track_id]

                for i in range(1, len(pts)):
                    if pts[i - 1] is None or pts[i] is None:
                        continue

                    thickness = max(1, int(5 - i / 20))

                    cv2.line(
                        frame,
                        pts[i - 1],
                        pts[i],
                        color,  # use team color
                        thickness
                    )

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with track ID
            label_text = f"{display_label}"
            if track_id is not None:
                label_text += f"#{track_id}"
            label_text += f" {conf:.2f}"

            cv2.putText(
                frame,
                label_text,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        
        for p in players:
            if len(p) >= 9 and not p[5]:  # not referee
                track_id = p[7]
                team_id = p[8]
                
                if track_id in player_positions:
                    last_pos = player_positions[track_id][-1]
                    if frame_idx % 5 == 0:
                        if team_id == 0:
                            team1_positions.append(last_pos)
                        elif team_id == 1:
                            team2_positions.append(last_pos)

       

    

        for x1, y1, x2, y2, conf, track_id in balls:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BALL, 2)
            cv2.putText(
                frame,
                f"ball{'' if track_id is None else f'#{track_id}'} {conf:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_BALL,
                2,
            )

        # Possession: assign to nearest non-ref player (with team id) near the ball.
        owner_team = None
        owner_track_id = None
        if balls and players:
            ball_best = max(balls, key=lambda b: b[4])
            bx = 0.5 * (ball_best[0] + ball_best[2])
            by = 0.5 * (ball_best[1] + ball_best[3])
            best_dist = float("inf")
            for p in players:
                if p[5] or len(p) < 9:
                    continue
                px = 0.5 * (p[0] + p[2])
                py = float(p[3])  # foot-center y
                dist = float(np.hypot(px - bx, py - by))
                if dist < best_dist:
                    best_dist = dist
                    owner_team = int(p[8])
                    owner_track_id = p[7]
            if best_dist > float(args.possession_radius):
                owner_team = None
                owner_track_id = None

        # Smoothed runtime FPS for operator feedback + backend stats.
        now = time.perf_counter()
        dt = max(1e-6, now - t_prev)
        inst_fps = 1.0 / dt
        live_fps = inst_fps if live_fps == 0.0 else (0.92 * live_fps + 0.08 * inst_fps)
        t_prev = now
        # Debounce ownership switches to reduce flicker between close players.
        if owner_track_id == current_owner_track_id:
            pending_owner_team = None
            pending_owner_track_id = None
            pending_owner_frames = 0
        else:
            if owner_track_id == pending_owner_track_id:
                pending_owner_frames += 1
            else:
                pending_owner_team = owner_team
                pending_owner_track_id = owner_track_id
                pending_owner_frames = 1
            if pending_owner_frames >= switch_frames:
                if (
                    previous_owner_team == pending_owner_team
                    and previous_owner_track_id != pending_owner_track_id
                    and previous_owner_track_id is not None
                ):
                    passes += 1

                # =========================
                # TOUCH COUNT (existing)
                # =========================
                if pending_owner_track_id is not None:
                    player_touch_counts[pending_owner_track_id] = (
                        player_touch_counts.get(pending_owner_track_id, 0) + 1
                    )
                    if pending_owner_team in (0, 1):
                        team_touch_counts[pending_owner_team] += 1

                # =========================
                # UPDATE PREVIOUS OWNER
                # =========================
                previous_owner_team = pending_owner_team
                previous_owner_track_id = pending_owner_track_id

                # =========================
                # CURRENT OWNER UPDATE
                # =========================
                current_owner_team = pending_owner_team
                current_owner_track_id = pending_owner_track_id

                pending_owner_team = None
                pending_owner_track_id = None
                pending_owner_frames = 0

        if current_owner_team in (0, 1):
            team_pos_seconds[current_owner_team] += dt

        player_distance = {}

        for track_id, positions in player_positions.items():
            dist = 0
            for i in range(1, len(positions)):
                x1, y1 = positions[i-1]
                x2, y2 = positions[i]
                dist += np.hypot(x2 - x1, y2 - y1)

            player_distance[track_id] = dist

        if frame_idx % stats_every == 0:
            team1_count = sum(1 for p in players if (not p[5]) and len(p) >= 9 and p[8] == 0)
            team2_count = sum(1 for p in players if (not p[5]) and len(p) >= 9 and p[8] == 1)
            referee_count = sum(1 for p in players if p[5])
            players_count = sum(1 for p in players if not p[5])
            tracked_player_ids = sorted({int(p[7]) for p in players if p[7] is not None})
            tracked_ball_ids = sorted({int(b[5]) for b in balls if b[5] is not None})
            poss_total = team_pos_seconds[0] + team_pos_seconds[1]
            team1_pct = (100.0 * team_pos_seconds[0] / poss_total) if poss_total > 0 else 0.0
            team2_pct = (100.0 * team_pos_seconds[1] / poss_total) if poss_total > 0 else 0.0
            payload = {
                "timestamp": time.time(),
                "total_passes": passes,
                "frame": frame_idx,
                "fps": round(float(live_fps), 2),
                "players_detected": players_count,
                "team1_players": team1_count,
                "team2_players": team2_count,
                "referees_detected": referee_count,
                "ball_detected": bool(balls),
                "tracked_player_ids": tracked_player_ids,
                "tracked_ball_ids": tracked_ball_ids,
                "possession_team": (
                    None
                    if current_owner_team is None
                    else ("team1" if current_owner_team == 0 else "team2")
                ),
                "ball_owner_track_id": current_owner_track_id,
                "team1_possession_pct": round(team1_pct, 2),
                "team2_possession_pct": round(team2_pct, 2),
                "team1_touches": int(team_touch_counts[0]),
                "team2_touches": int(team_touch_counts[1]),
                "player_touch_counts": {str(k): int(v) for k, v in player_touch_counts.items()},
                "possession_switch_frames": switch_frames,
                "player_distance": {str(k): float(v) for k, v in player_distance.items()},
            }
            write_json_atomic(args.stats_out, payload)



        if show_preview:
            cv2.imshow(PREVIEW_WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("Exiting video...")
                break
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Generating heatmaps...")


    if team1_positions:
        generate_pitch_heatmap(team1_positions, pitch, "team1_pitch.png", frame_w, frame_h)

    if team2_positions:
        generate_pitch_heatmap(team2_positions, pitch, "team2_pitch.png", frame_w, frame_h)

    print("Heatmaps saved!")


if __name__ == "__main__":
    main()
