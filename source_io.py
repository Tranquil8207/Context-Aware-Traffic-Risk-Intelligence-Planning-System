"""Load ingested sources written by the UI (data/sources/<id>/ingest.json).

Detection and the CLI calibrators take a source_id, a path, or — if you
pass nothing — the most recently ingested source. There is no hardcoded
video or place fallback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "sources"

# Tunables for event identification. The UI step after lane boxing edits
# these; objectdetection.py reads the saved copy off the ingest record.
EVENT_DEFAULTS: dict[str, float] = {
    "CONFIDENCE": 0.35,
    "SPEED_SMOOTHING_FRAMES": 10,
    "MAX_REASONABLE_SPEED_KMH": 180.0,
    "WRONG_WAY_ANGLE_DEG": 150.0,
    "WRONG_WAY_DWELL_S": 4.0,
    "SPEEDING_OVER_WINDOW_S": 0.5,
    "SPEEDING_UNDER_WINDOW_S": 1.0,
    "HARSH_BRAKE_DROP_KMH": 7.0,  # MDPI harsh-braking paper: 6 ft/s2 (~0.2g) rounded up
    "HARSH_BRAKE_WINDOW_S": 1.0,
    "HARSH_BRAKE_MIN_SPEED_KMH": 20.0,
    "NEAR_MISS_GAP_M": 2.0,
    "NEAR_MISS_MIN_FRAMES": 3,
    "WEAVE_VLAT_LIM_KMH": 6.0,
    "WEAVE_WINDOW_FRAMES": 5,
}

INT_EVENT_KEYS = {
    "SPEED_SMOOTHING_FRAMES",
    "NEAR_MISS_MIN_FRAMES",
    "WEAVE_WINDOW_FRAMES",
}

# MoRTH "Road Accidents in India 2024" report, persons-killed share by
# traffic rule violation (percentage points, e.g. 71.2 not 0.712 -- the
# risk_score.py weight formula ln(1+s_k) only meaningfully dampens a
# dominant cause like speeding at this scale; at fraction scale (0.712)
# the log transform barely differs from the raw linear share).
# lane_cut has no separate MoRTH category (folded into "Others") and is
# left out rather than guessed -- this project's Stage 2 also never emits
# lane_cut incidents, so its weight would never multiply anything nonzero
# anyway.
RISK_S_K_DEFAULTS: dict[str, float] = {
    "speeding": 71.2,
    "wrong_way": 5.4,   # MoRTH category: "Driving on wrong side"
    "red_light": 0.9,   # MoRTH category: "Jumping red light"
}

# Step 8 blend weights (V/P/E/C -> R). NOT derived from MoRTH -- there is
# no published source for how these four terms should combine. Calibrated
# by reverse-engineering a uniform scale factor against make_test_db_multi's
# four synthetic scenarios (QUIET/BUSY/EMPTY/SCHOOL) so their expected bands
# (cold/warm/cold/hot) actually land where that fixture's own comments say
# they should -- not an authoritative source, just a starting point tuned
# to that test data. Still meant to be refined further, e.g. via the +-20%
# sensitivity sweep from the innovation layer.
RISK_ABCD_DEFAULTS: dict[str, float] = {
    "a": 4.0,
    "b": 4.0,
    "c": 4.0,
    "d": 4.0,
}


def merge_events(
    saved: Optional[dict[str, Any]] = None,
    v_min_kmh: Optional[float] = None,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(EVENT_DEFAULTS)
    if v_min_kmh is not None:
        merged["V_MIN_KMH"] = v_min_kmh
    if saved:
        for key, value in saved.items():
            if key in EVENT_DEFAULTS or key == "V_MIN_KMH":
                merged[key] = value
    for key in INT_EVENT_KEYS:
        if merged.get(key) is not None:
            merged[key] = int(merged[key])
    return merged


def read_ingest(source_dir: Path) -> Optional[dict[str, Any]]:
    path = source_dir / "ingest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def list_source_records() -> list[dict[str, Any]]:
    if not DATA_DIR.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in DATA_DIR.iterdir():
        if not child.is_dir():
            continue
        record = read_ingest(child)
        if record:
            rows.append(record)
    rows.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return rows


def video_file_for(record: dict[str, Any]) -> Path:
    rel = record.get("video_path")
    if not rel:
        raise FileNotFoundError(f"Source {record.get('source_id')!r} has no video_path")
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    return path


def resolve_source(arg: Optional[str] = None) -> dict[str, Any]:
    """Return an ingest record.

    `arg` may be a source_id, a path to a source folder, ingest.json, or a
    video file. With no arg, the latest ingested source is used.
    """
    if arg:
        as_id = DATA_DIR / arg
        if as_id.is_dir() and (as_id / "ingest.json").is_file():
            record = read_ingest(as_id)
            if record:
                return record

        path = Path(arg)
        if path.is_file() and path.name == "ingest.json":
            record = read_ingest(path.parent)
            if record:
                return record
        if path.is_dir() and (path / "ingest.json").is_file():
            record = read_ingest(path)
            if record:
                return record
        if path.is_file():
            parent = path.parent
            record = read_ingest(parent)
            if record:
                return record
            raise FileNotFoundError(
                f"{path} is not an ingested UI source. Ingest the video at "
                "http://localhost:8000 first."
            )
        raise FileNotFoundError(
            f"No ingested source named {arg!r}. Ingest a video in the UI first."
        )

    records = list_source_records()
    if not records:
        raise FileNotFoundError(
            "No ingested sources in data/sources. Open http://localhost:8000 "
            "and ingest a video first."
        )
    return records[0]
