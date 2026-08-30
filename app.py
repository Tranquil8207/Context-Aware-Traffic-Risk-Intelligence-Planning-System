"""XFLR5-style web UI: ingest → homography → lanes.

Run:
    python app.py
    # then open http://localhost:8000
"""

from __future__ import annotations

import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from risk_score import combine_risk, compute_metrics, save_risk_row
from source_io import EVENT_DEFAULTS, RISK_ABCD_DEFAULTS, RISK_S_K_DEFAULTS, merge_events

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
DATA_DIR = ROOT / "data" / "sources"
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,180}$")

app = FastAPI(title="Traffic Risk Intelligence", version="0.1")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _opt_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return float(text)


def _flag(value: Optional[str]) -> int:
    if value is None:
        return 0
    return 1 if str(value).strip().lower() in {"1", "true", "on", "yes"} else 0


def _safe_source_id(raw: str, fallback: str) -> str:
    candidate = (raw or "").strip() or fallback
    candidate = candidate.replace(" ", "_")
    if not SOURCE_ID_RE.match(candidate):
        candidate = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._-")
    if not candidate:
        candidate = "source"
    return candidate[:180]


def _unique_source_id(source_id: str) -> str:
    if not (DATA_DIR / source_id).exists():
        return source_id
    n = 2
    while (DATA_DIR / f"{source_id}_{n}").exists():
        n += 1
    return f"{source_id}_{n}"


def _source_dir(source_id: str) -> Path:
    return DATA_DIR / source_id


def _write_ingest(source_id: str, record: dict[str, Any]) -> None:
    path = _source_dir(source_id) / "ingest.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _probe_video(path: Path) -> dict[str, Any]:
    media: dict[str, Any] = {
        "width": None,
        "height": None,
        "fps": None,
        "frame_count": None,
        "duration_s": None,
    }
    try:
        import cv2
    except Exception:
        return media

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return media
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    media["width"] = width or None
    media["height"] = height or None
    media["fps"] = round(fps, 3) if fps else None
    media["frame_count"] = frames or None
    if fps and frames:
        media["duration_s"] = round(frames / fps, 3)
    return media


def _read_ingest(source_dir: Path) -> Optional[dict[str, Any]]:
    path = source_dir / "ingest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _require_source(source_id: str) -> dict[str, Any]:
    record = _read_ingest(_source_dir(source_id))
    if not record:
        raise HTTPException(status_code=404, detail="Source not found")
    return record


def _video_path(record: dict[str, Any]) -> Path:
    path = ROOT / record["video_path"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video file missing")
    return path


def _list_sources() -> list[dict[str, Any]]:
    if not DATA_DIR.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in sorted(DATA_DIR.iterdir()):
        if child.is_dir():
            record = _read_ingest(child)
            if record:
                rows.append(record)
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def _persist_supabase(record: dict[str, Any]) -> dict[str, Any]:
    try:
        from supabase_client import get_client

        client = get_client()
        place = record["place"]
        processing = record.get("processing") or {}
        place_row = {
            "lat": place.get("lat"),
            "lng": place.get("lng"),
            "road_kind": place.get("road_kind"),
            "speed_limit_kmh": place.get("speed_limit_kmh"),
            "ref_length_m": processing.get("lane_width_m"),
            "ref_points_json": (record.get("homography") or {}).get("src"),
            "stop_line_json": None,
            "legal_heading": None,
            "lanes_json": record.get("lanes"),
            "has_signal": place.get("has_signal", 0),
            "city_prior": place.get("city_prior"),
            "is_night": place.get("is_night", 0),
            "is_rain": place.get("is_rain", 0),
            "is_rush": place.get("is_rush", 0),
        }
        place_resp = client.table("places").insert(place_row).execute()
        place_id = place_resp.data[0]["place_id"]
        client.table("cleaned_inputs").insert(
            {
                "source_id": record["source_id"],
                "kind": record.get("kind") or "video",
                "place_id": place_id,
                "file_name": record.get("file_name"),
                "clock_start": record.get("clock_start"),
                "clock_note": record.get("clock_note"),
                "notes": record.get("notes"),
            }
        ).execute()
        return {"ok": True, "place_id": place_id}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _build_homography(src: list, lane_width_m: float, reference_distance_m: float):
    import cv2
    import numpy as np

    src_pts = np.float32(src)
    dst_pts = np.float32(
        [
            [0.0, 0.0],
            [lane_width_m, 0.0],
            [0.0, reference_distance_m],
            [lane_width_m, reference_distance_m],
        ]
    )
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def _pixel_to_world(H, x: float, y: float) -> tuple[float, float]:
    import cv2
    import numpy as np

    point = np.array([[[float(x), float(y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def _heading_deg(H, start: list, end: list) -> float:
    x1, y1 = _pixel_to_world(H, start[0], start[1])
    x2, y2 = _pixel_to_world(H, end[0], end[1])
    return math.degrees(math.atan2(x2 - x1, y2 - y1))


def _save_calibration_files(source_id: str, record: dict[str, Any]) -> None:
    payload = {
        "homography": record.get("homography") or {},
        "lanes": record.get("lanes") or [],
    }
    (_source_dir(source_id) / "calibration.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


class HomographyIn(BaseModel):
    src: list[list[float]] = Field(..., min_length=4, max_length=4)
    lane_width_m: float
    reference_distance_m: float
    frame_index: int = 0


class LaneIn(BaseModel):
    polygon: list[list[float]] = Field(..., min_length=3)
    arrow: Optional[list[list[float]]] = None


class LanesIn(BaseModel):
    lanes: list[LaneIn] = Field(..., min_length=1)
    frame_index: int = 0


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/sources")
def list_sources() -> dict[str, Any]:
    return {"sources": _list_sources()}


@app.get("/api/sources/{source_id}")
def get_source(source_id: str) -> dict[str, Any]:
    return _require_source(source_id)


@app.get("/api/sources/{source_id}/video")
def get_video(source_id: str) -> FileResponse:
    record = _require_source(source_id)
    path = _video_path(record)
    return FileResponse(path, filename=record.get("file_name") or path.name)


@app.get("/api/sources/{source_id}/frame")
def get_frame(source_id: str, index: int = 0) -> Response:
    import cv2

    record = _require_source(source_id)
    path = _video_path(record)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Could not open video")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        cap.release()
        raise HTTPException(status_code=500, detail="Video has no frames")
    index = max(0, min(int(index), count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(status_code=404, detail="Could not read frame")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode frame")
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Frame-Index": str(index),
            "X-Frame-Count": str(count),
            "X-Frame-Width": str(int(frame.shape[1])),
            "X-Frame-Height": str(int(frame.shape[0])),
            "Cache-Control": "no-store",
        },
    )


@app.post("/api/ingest")
async def ingest(
    video: UploadFile = File(...),
    source_id: str = Form(""),
    lat: str = Form(""),
    lng: str = Form(""),
    road_kind: str = Form("expressway"),
    speed_limit_kmh: str = Form("80"),
    v_min_kmh: str = Form(""),
    has_signal: str = Form("0"),
    is_night: str = Form("0"),
    is_rain: str = Form("0"),
    is_rush: str = Form("0"),
    city_prior: str = Form("1.0"),
) -> dict[str, Any]:
    original_name = Path(video.filename or "upload.mp4").name
    suffix = Path(original_name).suffix.lower() or ".mp4"
    if suffix not in VIDEO_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video type '{suffix}'. Use mp4, avi, mkv, mov, webm.",
        )

    source_id = _unique_source_id(_safe_source_id(source_id, Path(original_name).stem))
    source_dir = _source_dir(source_id)
    source_dir.mkdir(parents=True, exist_ok=True)
    dest = source_dir / original_name
    try:
        with dest.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)
    finally:
        await video.close()

    if dest.stat().st_size == 0:
        shutil.rmtree(source_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    record: dict[str, Any] = {
        "source_id": source_id,
        "kind": "video",
        "file_name": original_name,
        "clock_start": None,
        "clock_note": None,
        "notes": None,
        "video_path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "created_at": _utc_now(),
        "stage": "homography",
        "media": _probe_video(dest),
        "place": {
            "lat": _opt_float(lat),
            "lng": _opt_float(lng),
            "road_kind": road_kind.strip() or "expressway",
            "speed_limit_kmh": _opt_float(speed_limit_kmh),
            "has_signal": _flag(has_signal),
            "city_prior": _opt_float(city_prior),
            "is_night": _flag(is_night),
            "is_rain": _flag(is_rain),
            "is_rush": _flag(is_rush),
        },
        "processing": {
            "v_min_kmh": _opt_float(v_min_kmh),
        },
        "homography": None,
        "lanes": None,
        "place_id": None,
    }

    supabase = _persist_supabase(record)
    if supabase.get("ok"):
        record["place_id"] = supabase.get("place_id")
    record["supabase"] = supabase
    _write_ingest(source_id, record)
    return record


@app.post("/api/sources/{source_id}/homography")
def save_homography(source_id: str, body: HomographyIn) -> dict[str, Any]:
    record = _require_source(source_id)
    if any(len(pt) != 2 for pt in body.src):
        raise HTTPException(status_code=400, detail="Each point must be [x, y]")
    if body.lane_width_m <= 0 or body.reference_distance_m <= 0:
        raise HTTPException(status_code=400, detail="Measurements must be > 0")

    try:
        H = _build_homography(body.src, body.lane_width_m, body.reference_distance_m)
        if H is None:
            raise ValueError("findHomography returned None")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not build homography: {exc}") from exc

    record["homography"] = {
        "src": [[int(round(x)), int(round(y))] for x, y in body.src],
        "lane_width_m": float(body.lane_width_m),
        "reference_distance_m": float(body.reference_distance_m),
        "frame_index": int(body.frame_index),
    }
    record["processing"] = {
        **(record.get("processing") or {}),
        "lane_width_m": float(body.lane_width_m),
        "reference_distance_m": float(body.reference_distance_m),
    }
    record["stage"] = "lanes"
    _write_ingest(source_id, record)
    _save_calibration_files(source_id, record)
    return record


@app.post("/api/sources/{source_id}/lanes")
def save_lanes(source_id: str, body: LanesIn) -> dict[str, Any]:
    record = _require_source(source_id)
    homo = record.get("homography") or {}
    H = None
    if homo.get("src") and homo.get("lane_width_m") and homo.get("reference_distance_m"):
        H = _build_homography(
            homo["src"],
            float(homo["lane_width_m"]),
            float(homo["reference_distance_m"]),
        )

    lanes = []
    for i, lane in enumerate(body.lanes, start=1):
        heading = None
        arrow = lane.arrow
        if H is not None and arrow and len(arrow) == 2:
            heading = _heading_deg(H, arrow[0], arrow[1])
        lanes.append(
            {
                "lane": i,
                "polygon": [[int(round(x)), int(round(y))] for x, y in lane.polygon],
                "heading": heading,
                "arrow": [[int(round(x)), int(round(y))] for x, y in arrow] if arrow and len(arrow) == 2 else None,
                "frame_index": int(body.frame_index),
            }
        )

    record["lanes"] = [
        {"lane": item["lane"], "polygon": item["polygon"], "heading": item["heading"]}
        for item in lanes
    ]
    record["lane_arrows"] = [item.get("arrow") for item in lanes]
    record["stage"] = "events"
    if not record.get("events"):
        record["events"] = merge_events(
            None,
            (record.get("processing") or {}).get("v_min_kmh"),
        )
    _write_ingest(source_id, record)
    _save_calibration_files(source_id, record)
    return record


class EventsIn(BaseModel):
    CONFIDENCE: Optional[float] = None
    SPEED_SMOOTHING_FRAMES: Optional[float] = None
    MAX_REASONABLE_SPEED_KMH: Optional[float] = None
    WRONG_WAY_ANGLE_DEG: Optional[float] = None
    WRONG_WAY_DWELL_S: Optional[float] = None
    SPEEDING_OVER_WINDOW_S: Optional[float] = None
    SPEEDING_UNDER_WINDOW_S: Optional[float] = None
    HARSH_BRAKE_DROP_KMH: Optional[float] = None
    HARSH_BRAKE_WINDOW_S: Optional[float] = None
    HARSH_BRAKE_MIN_SPEED_KMH: Optional[float] = None
    NEAR_MISS_GAP_M: Optional[float] = None
    NEAR_MISS_MIN_FRAMES: Optional[float] = None
    WEAVE_VLAT_LIM_KMH: Optional[float] = None
    WEAVE_WINDOW_FRAMES: Optional[float] = None
    V_MIN_KMH: Optional[float] = None


@app.get("/api/event-defaults")
def event_defaults() -> dict[str, Any]:
    return {"defaults": EVENT_DEFAULTS}


@app.get("/api/risk-defaults")
def risk_defaults() -> dict[str, Any]:
    return {"s_k": RISK_S_K_DEFAULTS, "weights": RISK_ABCD_DEFAULTS}


@app.post("/api/sources/{source_id}/events")
def save_events(source_id: str, body: EventsIn) -> dict[str, Any]:
    record = _require_source(source_id)
    raw = body.model_dump()
    incoming = {k: v for k, v in raw.items() if v is not None}
    events = merge_events(incoming, incoming.get("V_MIN_KMH"))
    if raw.get("V_MIN_KMH") is None:
        events.pop("V_MIN_KMH", None)
    record["events"] = events
    record["processing"] = {
        **(record.get("processing") or {}),
        "v_min_kmh": events.get("V_MIN_KMH"),
    }
    record["stage"] = "events"
    _write_ingest(source_id, record)
    return record


class RiskMetricsIn(BaseModel):
    s_k: dict[str, float]


class RiskIn(BaseModel):
    s_k: dict[str, float]
    a: float
    b: float
    c: float
    d: float
    window: str = "this_clip"


def _risk_preconditions(record: dict[str, Any]) -> tuple[int, float]:
    place_id = record.get("place_id")
    if place_id is None:
        raise HTTPException(status_code=400, detail="No place_id yet -- run identify first.")

    duration_s = (record.get("media") or {}).get("duration_s")
    if not duration_s:
        raise HTTPException(status_code=400, detail="No clip duration recorded for this source.")

    return place_id, duration_s


@app.get("/api/sources/{source_id}/risk")
def get_risk(source_id: str) -> dict[str, Any]:
    record = _require_source(source_id)
    return {"risk": record.get("risk")}


@app.post("/api/sources/{source_id}/risk/metrics")
def get_risk_metrics(source_id: str, body: RiskMetricsIn) -> dict[str, Any]:
    """Steps 1-7 only -- V/P/E/C, independent of a/b/c/d. Call this once
    per s_k value; the UI should then recompute R live via combine_risk's
    formula (a*V + b*P + c*E + d*C, clamped 0-100) on every a/b/c/d slider
    change without calling this endpoint again."""
    record = _require_source(source_id)
    place_id, duration_s = _risk_preconditions(record)

    from supabase_client import get_client

    client = get_client()
    return compute_metrics(client, place_id, source_id, duration_s, body.s_k)


@app.post("/api/sources/{source_id}/risk")
def save_risk(source_id: str, body: RiskIn) -> dict[str, Any]:
    record = _require_source(source_id)
    place_id, duration_s = _risk_preconditions(record)

    from supabase_client import get_client

    client = get_client()
    try:
        metrics = compute_metrics(client, place_id, source_id, duration_s, body.s_k)
        combined = combine_risk(metrics, body.a, body.b, body.c, body.d)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not compute risk: {exc}") from exc

    row = {
        **combined,
        "place_id": place_id,
        "window": body.window,
    }
    try:
        saved = save_risk_row(client, place_id, body.window, combined)
        row.update(saved)
        record.pop("risk_db_error", None)
    except Exception as exc:
        record["risk_db_error"] = str(exc)

    record["risk"] = {
        k: v for k, v in row.items() if k not in {"R_raw", "db_warning"}
    }
    record["stage"] = "risk"
    if record.get("risk_db_error"):
        row["db_warning"] = record["risk_db_error"]
    _write_ingest(source_id, record)
    return row


@app.get("/api/sources/{source_id}/incidents")
def get_incidents(source_id: str) -> dict[str, Any]:
    _require_source(source_id)
    path = _source_dir(source_id) / "incidents.json"
    if not path.is_file():
        return {"incidents": []}
    try:
        return {"incidents": json.loads(path.read_text(encoding="utf-8"))}
    except json.JSONDecodeError:
        return {"incidents": []}


@app.post("/api/sources/{source_id}/identify")
def start_identify(source_id: str) -> dict[str, Any]:
    record = _require_source(source_id)
    if not record.get("lanes"):
        raise HTTPException(status_code=400, detail="Box lanes before running identification.")
    if not record.get("homography"):
        raise HTTPException(status_code=400, detail="Save homography before running identification.")

    import subprocess
    import sys
    import threading

    log_path = _source_dir(source_id) / "identify.log"
    record["run"] = {
        "status": "running",
        "started_at": _utc_now(),
        "log_path": str(log_path.relative_to(ROOT)).replace("\\", "/"),
    }
    _write_ingest(source_id, record)

    log_handle = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, "-u", str(ROOT / "objectdetection.py"), source_id],
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    record["run"]["pid"] = proc.pid
    _write_ingest(source_id, record)

    def _watch() -> None:
        code = proc.wait()
        log_handle.close()
        latest = _read_ingest(_source_dir(source_id)) or record
        latest["run"] = {
            **(latest.get("run") or {}),
            "status": "ok" if code == 0 else "failed",
            "returncode": code,
            "finished_at": _utc_now(),
        }
        if code == 0:
            latest["stage"] = "ready"
        _write_ingest(source_id, latest)

    threading.Thread(target=_watch, daemon=True).start()
    return record


@app.get("/api/sources/{source_id}/identify")
def identify_status(source_id: str) -> dict[str, Any]:
    record = _require_source(source_id)
    run = record.get("run") or {}
    log_path = _source_dir(source_id) / "identify.log"
    tail = ""
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        tail = "\n".join(text.splitlines()[-40:])
    incidents_path = _source_dir(source_id) / "incidents.json"
    incidents = []
    if incidents_path.is_file():
        try:
            incidents = json.loads(incidents_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            incidents = []
    return {"run": run, "log_tail": tail, "incidents": incidents}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
