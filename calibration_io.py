"""Shared load/save helpers for calibration.json.

Used by calibrate_homography.py, calibrate_lanes.py, and objectdetection.py
so all three agree on one file and schema instead of values being manually
copy-pasted between them.

Schema:
{
    "homography": {
        "src": [[x, y], [x, y], [x, y], [x, y]],
        "lane_width_m": 10.5,
        "reference_distance_m": 120.0
    },
    "lanes": [
        {"lane": 1, "polygon": [[x, y], ...]},
        ...
    ]
}
"""

import json
from pathlib import Path

CALIBRATION_PATH = Path(__file__).resolve().parent / "calibration.json"


def load_calibration() -> dict:
    if not CALIBRATION_PATH.exists():
        return {}
    return json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))


def save_calibration(data: dict) -> None:
    CALIBRATION_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_calibration(section: str, value) -> None:
    """Merge `value` into calibration.json under `section`, preserving
    whatever else is already saved (e.g. saving lanes doesn't wipe out an
    existing homography section, and vice versa)."""
    data = load_calibration()
    data[section] = value
    save_calibration(data)
