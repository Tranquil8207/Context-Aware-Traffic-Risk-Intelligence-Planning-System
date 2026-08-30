import json
import math
import os
import sys
from collections import deque

import cv2
import numpy as np
from supabase import Client
from ultralytics import YOLO

from source_io import merge_events, resolve_source, video_file_for
from supabase_client import get_client


# ============================================================
# CONFIGURATION
# ============================================================
#
# Video, place, homography, and lanes come from a UI ingest record
# (data/sources/<id>/ingest.json). apply_source() fills the names below
# at the start of main(). Detector weights stay here because they are
# not per-clip inputs.

OBJECT_MODEL_PATH = "yolo26m.pt"

CONFIDENCE = None
SPEED_SMOOTHING_FRAMES = None
MAX_REASONABLE_SPEED_KMH = None
MAX_REASONABLE_SPEED_MPS = None
WRONG_WAY_ANGLE_DEG = None
WRONG_WAY_DWELL_S = None
SPEEDING_OVER_WINDOW_S = None
SPEEDING_UNDER_WINDOW_S = None
HARSH_BRAKE_DROP_KMH = None
HARSH_BRAKE_WINDOW_S = None
HARSH_BRAKE_MIN_SPEED_KMH = None
NEAR_MISS_GAP_M = None
NEAR_MISS_MIN_FRAMES = None
WEAVE_VLAT_LIM_KMH = None
WEAVE_WINDOW_FRAMES = None

VIDEO_PATH = None
OUTPUT_VIDEO = None
SOURCE_ID = None
PLACE_ID = None
HOMOGRAPHY_SRC = None
LANE_WIDTH_M = None
REFERENCE_DISTANCE_M = None
CAMERA_HEIGHT_M = None
V_MIN_KMH = None
LAT = None
LNG = None
ROAD_KIND = None
SPEED_LIMIT_KMH = None
REF_LENGTH_M = None
REF_POINTS = None
STOP_LINE = None
LEGAL_HEADING = None
LANES = None
HAS_SIGNAL = 0
CITY_PRIOR = None
IS_NIGHT = 0
IS_RAIN = 0
IS_RUSH = 0
CLOCK_START = None
CLOCK_NOTE = None
NOTES = None


def apply_source(record: dict) -> None:
    """Copy one UI ingest record into the module-level names used below."""
    global VIDEO_PATH, OUTPUT_VIDEO, SOURCE_ID, PLACE_ID
    global HOMOGRAPHY_SRC, LANE_WIDTH_M, REFERENCE_DISTANCE_M, CAMERA_HEIGHT_M
    global V_MIN_KMH, LAT, LNG, ROAD_KIND, SPEED_LIMIT_KMH, REF_LENGTH_M
    global REF_POINTS, STOP_LINE, LEGAL_HEADING, LANES
    global HAS_SIGNAL, CITY_PRIOR, IS_NIGHT, IS_RAIN, IS_RUSH
    global CLOCK_START, CLOCK_NOTE, NOTES
    global CONFIDENCE, SPEED_SMOOTHING_FRAMES, MAX_REASONABLE_SPEED_KMH, MAX_REASONABLE_SPEED_MPS
    global WRONG_WAY_ANGLE_DEG, WRONG_WAY_DWELL_S
    global SPEEDING_OVER_WINDOW_S, SPEEDING_UNDER_WINDOW_S
    global HARSH_BRAKE_DROP_KMH, HARSH_BRAKE_WINDOW_S, HARSH_BRAKE_MIN_SPEED_KMH
    global NEAR_MISS_GAP_M, NEAR_MISS_MIN_FRAMES
    global WEAVE_VLAT_LIM_KMH, WEAVE_WINDOW_FRAMES

    video_path = video_file_for(record)
    SOURCE_ID = record["source_id"]
    PLACE_ID = record.get("place_id")
    VIDEO_PATH = str(video_path)
    OUTPUT_VIDEO = str(video_path.parent / "stage1_lane_tracking.mp4")

    homo = record.get("homography") or {}
    HOMOGRAPHY_SRC = homo.get("src")
    LANE_WIDTH_M = homo.get("lane_width_m")
    REFERENCE_DISTANCE_M = homo.get("reference_distance_m")
    REF_POINTS = HOMOGRAPHY_SRC
    REF_LENGTH_M = LANE_WIDTH_M

    processing = record.get("processing") or {}
    CAMERA_HEIGHT_M = processing.get("camera_height_m")
    events = merge_events(record.get("events"), processing.get("v_min_kmh"))
    CONFIDENCE = events["CONFIDENCE"]
    SPEED_SMOOTHING_FRAMES = events["SPEED_SMOOTHING_FRAMES"]
    MAX_REASONABLE_SPEED_KMH = events["MAX_REASONABLE_SPEED_KMH"]
    MAX_REASONABLE_SPEED_MPS = MAX_REASONABLE_SPEED_KMH / 3.6
    WRONG_WAY_ANGLE_DEG = events["WRONG_WAY_ANGLE_DEG"]
    WRONG_WAY_DWELL_S = events["WRONG_WAY_DWELL_S"]
    SPEEDING_OVER_WINDOW_S = events["SPEEDING_OVER_WINDOW_S"]
    SPEEDING_UNDER_WINDOW_S = events["SPEEDING_UNDER_WINDOW_S"]
    HARSH_BRAKE_DROP_KMH = events["HARSH_BRAKE_DROP_KMH"]
    HARSH_BRAKE_WINDOW_S = events["HARSH_BRAKE_WINDOW_S"]
    HARSH_BRAKE_MIN_SPEED_KMH = events["HARSH_BRAKE_MIN_SPEED_KMH"]
    NEAR_MISS_GAP_M = events["NEAR_MISS_GAP_M"]
    NEAR_MISS_MIN_FRAMES = events["NEAR_MISS_MIN_FRAMES"]
    WEAVE_VLAT_LIM_KMH = events["WEAVE_VLAT_LIM_KMH"]
    WEAVE_WINDOW_FRAMES = events["WEAVE_WINDOW_FRAMES"]
    V_MIN_KMH = events.get("V_MIN_KMH")

    place = record.get("place") or {}
    LAT = place.get("lat")
    LNG = place.get("lng")
    ROAD_KIND = place.get("road_kind")
    SPEED_LIMIT_KMH = place.get("speed_limit_kmh")
    HAS_SIGNAL = place.get("has_signal") or 0
    CITY_PRIOR = place.get("city_prior")
    IS_NIGHT = place.get("is_night") or 0
    IS_RAIN = place.get("is_rain") or 0
    IS_RUSH = place.get("is_rush") or 0
    STOP_LINE = place.get("stop_line_json")
    LEGAL_HEADING = place.get("legal_heading")

    LANES = record.get("lanes") or []
    CLOCK_START = record.get("clock_start")
    CLOCK_NOTE = record.get("clock_note")
    NOTES = record.get("notes")


# ============================================================
# VEHICLE CLASSES
# ============================================================

CLASS_MAP = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# GLOBAL CALIBRATION
# ============================================================

HOMOGRAPHY = None


# ============================================================
# HOMOGRAPHY
# ============================================================

def create_homography():

    r"""
    Four image points (near-left, near-right, far-left, far-right) mapped
    to a known real-world rectangle on the road:

        near-left  = (0, 0)
        near-right = (LANE_WIDTH_M, 0)
        far-left   = (0, REFERENCE_DISTANCE_M)
        far-right  = (LANE_WIDTH_M, REFERENCE_DISTANCE_M)

    Points and metres come from the UI homography step (ingest.json).
    """

    if not HOMOGRAPHY_SRC or LANE_WIDTH_M is None or REFERENCE_DISTANCE_M is None:
        raise RuntimeError(
            "Homography inputs are missing. Finish step 2 in the UI first."
        )

    src = np.float32(HOMOGRAPHY_SRC)

    dst = np.float32([
        [0.0, 0.0],
        [LANE_WIDTH_M, 0.0],
        [0.0, REFERENCE_DISTANCE_M],
        [LANE_WIDTH_M, REFERENCE_DISTANCE_M],
    ])

    H, mask = cv2.findHomography(
        src,
        dst
    )

    print("\nHomography created.")
    print("Source points:")
    print(src)

    print("\nWorld points:")
    print(dst)

    return H


# ============================================================
# PIXEL -> ROAD COORDINATE
# ============================================================

def pixel_to_world(x, y):

    if HOMOGRAPHY is None:
        return None, None

    point = np.array(
        [[[float(x), float(y)]]],
        dtype=np.float32
    )

    transformed = cv2.perspectiveTransform(
        point,
        HOMOGRAPHY
    )

    X = float(
        transformed[0][0][0]
    )

    Y = float(
        transformed[0][0][1]
    )

    return X, Y


# ============================================================
# PIXEL -> LANE NUMBER
# ============================================================
#
# LANES is boxed per video in the UI (or calibrate_lanes.py) as a list of
# {"lane": n, "polygon": [[x, y], ...]} pixel-space polygons -- one per
# lane. LANE_POLYGONS caches the parsed numpy polygons; built once in main().

LANE_POLYGONS = None


def build_lane_polygons(lanes):
    if not lanes:
        return None

    return [
        (entry["lane"], np.array(entry["polygon"], dtype=np.int32))
        for entry in lanes
    ]


def get_lane_number(x, y):
    if LANE_POLYGONS is None:
        return None

    for lane_number, polygon in LANE_POLYGONS:
        if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
            return lane_number

    return None


# ============================================================
# LANE PROCESSING (manual boxing, from calibrate_lanes.py)
# ============================================================

LANE_POLYGON_COLORS = [
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
]


def draw_lane_polygons(frame, lane_polygons):
    """Outline manually-boxed lanes (from calibrate_lanes.py) for visual
    verification that they still line up with the road."""

    if not lane_polygons:
        return frame

    for i, (lane_number, polygon) in enumerate(lane_polygons):
        color = LANE_POLYGON_COLORS[i % len(LANE_POLYGON_COLORS)]

        cv2.polylines(frame, [polygon], True, color, 2)

        centroid = polygon.mean(axis=0).astype(int)
        cv2.putText(
            frame,
            f"Lane {lane_number}",
            (int(centroid[0]), int(centroid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return frame


# ============================================================
# VELOCITY
# ============================================================
#
# MAX_REASONABLE_SPEED_MPS is derived from MAX_REASONABLE_SPEED_KMH inside
# apply_source() (not here) -- computing it at module level would run
# before apply_source() ever sets a real value, dividing None by 3.6 and
# crashing on import.


def is_plausible_world_step(history, t, x, y):
    """Reject a position sample before it enters world_history, rather than
    only rejecting the speed computed from it afterward -- otherwise a
    single bad sample (detector glitch, homography extrapolation spike)
    keeps contaminating the velocity fit for as long as it stays in the
    window, not just the one frame it occurred on."""

    if not history:
        return True

    last_t, last_x, last_y = history[-1]
    dt = t - last_t

    if dt <= 0:
        return False

    dx = x - last_x
    dy = y - last_y

    return math.sqrt(dx * dx + dy * dy) / dt <= MAX_REASONABLE_SPEED_MPS


def calculate_velocity(
    track,
    current_time_s
):

    # A 2-point frame-to-frame difference amplifies any pixel/homography
    # jitter into velocity noise -- fitting a line through the whole
    # recent window (world_history holds up to 20 samples) and taking its
    # slope is far more stable, and degrades gracefully to the old 2-point
    # behavior when a track only has 2 samples so far.
    history = track["world_history"]

    if len(history) < 2:
        return 0.0, 0.0, 0.0

    times = np.array([sample[0] for sample in history])

    if times[-1] - times[0] <= 0:
        return 0.0, 0.0, 0.0

    xs = np.array([sample[1] for sample in history])
    ys = np.array([sample[2] for sample in history])

    vx = np.polyfit(times, xs, 1)[0]

    vy = np.polyfit(times, ys, 1)[0]

    speed = math.sqrt(
        vx * vx +
        vy * vy
    )

    return vx, vy, speed


def update_speed(
    track,
    timestamp_s
):

    vx, vy, speed = calculate_velocity(
        track,
        timestamp_s
    )

    speed_kmh = speed * 3.6

    # Remove physically impossible spikes
    if speed_kmh > MAX_REASONABLE_SPEED_KMH:

        return (
            track["speed_mps"],
            track["speed_kmh"],
            track["vx"],
            track["vy"]
        )

    track["speed_history"].append(
        speed
    )

    # Smoothed instantaneous velocity
    filtered_speed = np.mean(
        track["speed_history"]
    )

    track["speed_mps"] = filtered_speed

    track["speed_kmh"] = (
        filtered_speed * 3.6
    )

    track["vx"] = vx
    track["vy"] = vy

    return (
        track["speed_mps"],
        track["speed_kmh"],
        track["vx"],
        track["vy"]
    )


# ============================================================
# EVENT DETECTION
# ============================================================
#
# track["metrics"] is a deque of (t_s, vx, vy, speed_kmh, lane) appended
# once per frame with fresh, accepted data -- it's what speeding/harsh_brake
# look back through for their time windows. track["fired"] tracks which
# conditions are currently active so each rule emits once per episode
# rather than every frame it holds true.


def heading_degrees(vx, vy):
    return math.degrees(math.atan2(vx, vy))


def angle_diff_deg(a, b):
    return abs(((a - b + 180) % 360) - 180)


def lane_heading(lane):
    if lane is None:
        return None
    for entry in LANES:
        if entry.get("lane") == lane:
            return entry.get("heading")
    return None


def value_at_or_before(metrics, target_t):
    candidate = None
    for sample in metrics:
        if sample[0] <= target_t:
            candidate = sample
        else:
            break
    return candidate


def windowed_mean_speed(metrics, t_s, window_s):
    values = [m[3] for m in metrics if 0 <= t_s - m[0] <= window_s]
    if not values:
        return None
    return sum(values) / len(values)


def emit_incident(incidents, track_id, t_ms, event_type, source, meta):
    incidents.append({
        "track_id": track_id,
        "ts_ms": t_ms,
        "type": event_type,
        "source": source,
        "conf": 1.0,
        "meta_json": meta,
    })


def check_wrong_way(track, track_id, t_s, t_ms, vx, vy, lane, incidents):
    legal = lane_heading(lane)
    if legal is None:
        legal = LEGAL_HEADING
    if legal is None:
        track["wrong_way_since"] = None
        track["fired"]["wrong_way"] = False
        return

    heading = heading_degrees(vx, vy)
    diff = angle_diff_deg(heading, legal)

    if diff >= WRONG_WAY_ANGLE_DEG:
        if track["wrong_way_since"] is None:
            track["wrong_way_since"] = t_s
        elif (
            not track["fired"]["wrong_way"]
            and t_s - track["wrong_way_since"] >= WRONG_WAY_DWELL_S
        ):
            emit_incident(
                incidents, track_id, t_ms, "wrong_way", "video",
                {"heading_deg": heading, "legal_heading_deg": legal, "diff_deg": diff},
            )
            track["fired"]["wrong_way"] = True
    else:
        track["wrong_way_since"] = None
        track["fired"]["wrong_way"] = False


def check_speeding(track, track_id, t_s, t_ms, lane, incidents):
    mean_over = windowed_mean_speed(track["metrics"], t_s, SPEEDING_OVER_WINDOW_S)
    if SPEED_LIMIT_KMH is not None and mean_over is not None and mean_over > SPEED_LIMIT_KMH:
        if not track["fired"]["speeding_over"]:
            emit_incident(
                incidents, track_id, t_ms, "speeding", "video",
                {"kind": "over", "mean_speed_kmh": mean_over, "speed_limit_kmh": SPEED_LIMIT_KMH},
            )
            track["fired"]["speeding_over"] = True
    else:
        track["fired"]["speeding_over"] = False

    mean_under = windowed_mean_speed(track["metrics"], t_s, SPEEDING_UNDER_WINDOW_S)
    if (
        V_MIN_KMH is not None
        and mean_under is not None
        and mean_under < V_MIN_KMH
        and lane is not None
    ):
        if not track["fired"]["speeding_under"]:
            emit_incident(
                incidents, track_id, t_ms, "speeding", "video",
                {"kind": "under", "mean_speed_kmh": mean_under, "v_min_kmh": V_MIN_KMH},
            )
            track["fired"]["speeding_under"] = True
    else:
        track["fired"]["speeding_under"] = False


def check_harsh_brake(track, track_id, t_s, t_ms, vy, incidents):
    if t_s - track["first_t_s"] < HARSH_BRAKE_WINDOW_S:
        return  # new IDs don't count -- not enough history yet

    past = value_at_or_before(track["metrics"], t_s - HARSH_BRAKE_WINDOW_S)
    if past is None:
        track["fired"]["harsh_brake"] = False
        return

    past_speed_kmh = past[2] * 3.6
    if past_speed_kmh < HARSH_BRAKE_MIN_SPEED_KMH:
        # Too slow beforehand for a "drop" to mean anything (e.g. a
        # near-stationary car's vy jitter shouldn't read as braking).
        track["fired"]["harsh_brake"] = False
        return

    drop_kmh = (past[2] - float(vy)) * 3.6  # positive = slowing down (+Y is forward)

    if drop_kmh >= HARSH_BRAKE_DROP_KMH:
        if not track["fired"]["harsh_brake"]:
            emit_incident(
                incidents, track_id, t_ms, "harsh_brake", "inferred",
                {"drop_kmh": drop_kmh, "window_s": HARSH_BRAKE_WINDOW_S},
            )
            track["fired"]["harsh_brake"] = True
    else:
        track["fired"]["harsh_brake"] = False


def check_weave(track, track_id, t_ms, incidents):
    recent = list(track["metrics"])[-WEAVE_WINDOW_FRAMES:]
    if len(recent) < WEAVE_WINDOW_FRAMES:
        track["fired"]["weave"] = False
        return

    vlat_kmh = [m[1] * 3.6 for m in recent]
    sustained = all(abs(v) > WEAVE_VLAT_LIM_KMH for v in vlat_kmh)

    signs = [v > 0 for v in vlat_kmh if abs(v) > WEAVE_VLAT_LIM_KMH]
    sign_flip = any(a != b for a, b in zip(signs, signs[1:]))

    if sustained or sign_flip:
        if not track["fired"]["weave"]:
            emit_incident(
                incidents, track_id, t_ms, "weave", "inferred",
                {"vlat_kmh": vlat_kmh[-1], "sustained": sustained, "sign_flip": sign_flip},
            )
            track["fired"]["weave"] = True
    else:
        track["fired"]["weave"] = False


def check_near_miss(frame_positions, tracks, t_ms, incidents, streaks):
    ids = list(frame_positions.keys())

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            xa, ya = frame_positions[a]
            xb, yb = frame_positions[b]
            gap = math.hypot(xa - xb, ya - yb)
            key = (a, b) if a < b else (b, a)

            if gap < NEAR_MISS_GAP_M:
                streaks[key] = streaks.get(key, 0) + 1
                if streaks[key] > NEAR_MISS_MIN_FRAMES:
                    tracks[a]["fired"]["near_miss"] = True
                    tracks[b]["fired"]["near_miss"] = True
                if streaks[key] == NEAR_MISS_MIN_FRAMES + 1:
                    emit_incident(
                        incidents, a, t_ms, "near_miss", "inferred",
                        {"other_track_id": int(b), "gap_m": gap},
                    )
            else:
                streaks[key] = 0
                tracks[a]["fired"]["near_miss"] = False
                tracks[b]["fired"]["near_miss"] = False


# ============================================================
# DATABASE
# ============================================================

def create_place(client: Client):

    row = {

        "lat": LAT,

        "lng": LNG,

        "road_kind": ROAD_KIND,

        "speed_limit_kmh":
            SPEED_LIMIT_KMH,

        "ref_length_m":
            REF_LENGTH_M,

        "ref_points_json":
            REF_POINTS,

        "stop_line_json":
            STOP_LINE,

        "legal_heading":
            LEGAL_HEADING,

        "lanes_json":
            LANES,

        "has_signal":
            HAS_SIGNAL,

        "city_prior":
            CITY_PRIOR,

        "is_night":
            IS_NIGHT,

        "is_rain":
            IS_RAIN,

        "is_rush":
            IS_RUSH,
    }

    response = (
        client
        .table("places")
        .insert(row)
        .execute()
    )

    return response.data[0]["place_id"]


def create_input(
    client,
    place_id
):

    source_id = SOURCE_ID or os.path.basename(VIDEO_PATH)

    row = {

        "source_id":
            source_id,

        "kind":
            "video",

        "place_id":
            place_id,

        "file_name":
            os.path.basename(
                VIDEO_PATH
            ),

        "clock_start":
            CLOCK_START,

        "clock_note":
            CLOCK_NOTE,

        "notes":
            NOTES,
    }

    (
        client
        .table("cleaned_inputs")
        .insert(row)
        .execute()
    )

    return source_id


DB_INCIDENT_TYPES = {
    "wrong_way", "lane_cut", "red_light",
    "speeding", "near_miss", "harsh_brake", "weave",
}
DB_INCIDENT_SOURCES = {"video", "inferred"}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    return value


def _ensure_cleaned_input(client, source_id, place_id):
    existing = (
        client.table("cleaned_inputs")
        .select("source_id")
        .eq("source_id", source_id)
        .execute()
    )
    if existing.data:
        return
    create_input(client, place_id)


def save_tracks(
    client,
    tracks,
    source_id,
    place_id,
    keep_ids=None,
):

    keep_ids = {int(i) for i in (keep_ids or [])}
    rows = []

    for track_id, data in tracks.items():

        duration_s = (
            data["last_t_ms"]
            -
            data["first_t_ms"]
        ) / 1000.0

        tid = int(track_id)
        if duration_s < 0.4 and tid not in keep_ids:
            continue

        rows.append({
            "track_id": tid,
            "source_id": source_id,
            "place_id": place_id,
            "class": data["class"],
            "path_json": _jsonable(data["path"]),
            "mean_speed_kmh": float(data["speed_kmh"]),
            "object_risk": None,
        })

    saved_ids = {row["track_id"] for row in rows}
    if not rows:
        return saved_ids

    client.table("tracked_objects").upsert(
        rows, on_conflict="source_id,track_id"
    ).execute()
    return saved_ids


def save_incidents(client, incidents, source_id, place_id, saved_track_ids):

    if not incidents:
        return 0

    rows = []
    skipped = []
    for incident in incidents:
        itype = str(incident.get("type") or "")
        isource = str(incident.get("source") or "")
        tid = int(incident["track_id"])
        if itype not in DB_INCIDENT_TYPES:
            skipped.append((itype, "type not in schema"))
            continue
        if isource not in DB_INCIDENT_SOURCES:
            skipped.append((itype, f"bad source {isource}"))
            continue
        if tid not in saved_track_ids:
            skipped.append((itype, f"track {tid} not saved"))
            continue
        rows.append({
            "source_id": source_id,
            "place_id": place_id,
            "track_id": tid,
            "ts_ms": int(incident["ts_ms"]),
            "type": itype,
            "source": isource,
            "conf": float(incident.get("conf") if incident.get("conf") is not None else 1.0),
            "meta_json": _jsonable(incident.get("meta_json")),
        })

    if skipped:
        print(f"Skipped {len(skipped)} incident(s) before insert: {skipped[:8]}")

    if not rows:
        return 0

    try:
        client.table("incidents").insert(rows).execute()
        print(f"Inserted {len(rows)} incident(s) into incidents.")
        return len(rows)
    except Exception as batch_err:
        print(f"Batch incident insert failed ({batch_err}); inserting row by row.")

    ok = 0
    for row in rows:
        try:
            client.table("incidents").insert(row).execute()
            ok += 1
        except Exception as row_err:
            print(f"  skip {row['type']} track={row['track_id']}: {row_err}")
    print(f"Inserted {ok}/{len(rows)} incident(s) row-by-row.")
    return ok


# ============================================================
# MAIN
# ============================================================

def main():

    global HOMOGRAPHY
    global LANE_POLYGONS

    arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        record = resolve_source(arg)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    apply_source(record)
    print(f"\nSource: {SOURCE_ID}")
    print(f"Video:  {VIDEO_PATH}")

    if not HOMOGRAPHY_SRC or LANE_WIDTH_M is None or REFERENCE_DISTANCE_M is None:
        raise SystemExit(
            "Homography is missing for this source. Finish step 2 in the UI "
            "(http://localhost:8000) before running detection."
        )

    # --------------------------------------------------------
    # CHECK VIDEO
    # --------------------------------------------------------

    if not os.path.exists(
        VIDEO_PATH
    ):

        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    # --------------------------------------------------------
    # OBJECT MODEL
    # --------------------------------------------------------

    print(
        "\nLoading object detector..."
    )

    object_model = YOLO(
        OBJECT_MODEL_PATH
    )

    # --------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 30.0

    print(
        f"\nVideo: {width} x {height}"
    )

    print(
        f"FPS: {fps:.2f}"
    )

    # --------------------------------------------------------
    # HOMOGRAPHY
    # --------------------------------------------------------

    HOMOGRAPHY = create_homography()

    # --------------------------------------------------------
    # LANES
    # --------------------------------------------------------

    LANE_POLYGONS = build_lane_polygons(LANES)

    if LANE_POLYGONS is None:
        print(
            "\nLANES not set -- lane numbers will be recorded as None. "
            "Finish step 3 (lane boxing) in the UI first."
        )

    
    # --------------------------------------------------------
    # VIDEO OUTPUT
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():

        raise RuntimeError(
            "Could not create output video."
        )

    # --------------------------------------------------------
    # TRACK STORAGE
    # --------------------------------------------------------

    tracks = {}

    incidents = []
    near_miss_streaks = {}

    frame_index = 0

    # ========================================================
    # FRAME LOOP
    # ========================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        timestamp_s = (
            frame_index / fps
        )

        t_ms = int(
            timestamp_s * 1000
        )

        

        frame = draw_lane_polygons(
            frame,
            LANE_POLYGONS
        )

        # ====================================================
        # OBJECT DETECTION + BYTETRACK
        # ====================================================

        results = object_model.track(

            frame,

            persist=True,

            tracker="bytetrack.yaml",

            conf=CONFIDENCE,

            classes=[
                0,
                2,
                3,
                5,
                7
            ],

            verbose=False,
        )

        result = results[0]

        # Positions of every track that got a fresh, accepted sample this
        # exact frame -- used by check_near_miss() after the object loop.
        frame_positions = {}

        # ====================================================
        # OBJECTS
        # ====================================================

        if (

            result.boxes is not None

            and

            result.boxes.id is not None

        ):

            boxes = (
                result
                .boxes
                .xyxy
                .cpu()
                .numpy()
            )

            classes = (
                result
                .boxes
                .cls
                .cpu()
                .numpy()
            )

            confidences = (
                result
                .boxes
                .conf
                .cpu()
                .numpy()
            )

            ids = (
                result
                .boxes
                .id
                .cpu()
                .numpy()
                .astype(int)
            )

            for (

                bbox,
                cls,
                conf,
                track_id

            ) in zip(

                boxes,
                classes,
                confidences,
                ids

            ):

                if (
                    float(conf)
                    <
                    CONFIDENCE
                ):
                    continue

                class_id = int(cls)

                object_class = (
                    CLASS_MAP.get(
                        class_id,
                        "other"
                    )
                )

                x1, y1, x2, y2 = bbox

                # ------------------------------------------------
                # CONTACT POINT
                # ------------------------------------------------

                x = int(
                    (x1 + x2) / 2
                )

                y = int(y2)

                # ------------------------------------------------
                # IMAGE -> WORLD
                # ------------------------------------------------

                X, Y = pixel_to_world(
                    x,
                    y
                )

                # ------------------------------------------------
                # LANE NUMBER (pixel-space, from calibrate_lanes.py)
                # ------------------------------------------------

                lane_number = get_lane_number(x, y)

                # ------------------------------------------------
                # INITIALIZE TRACK
                # ------------------------------------------------

                if track_id not in tracks:

                    tracks[track_id] = {

                        "class":
                            object_class,

                        "first_t_ms":
                            t_ms,

                        "first_t_s":
                            t_ms / 1000.0,

                        "last_t_ms":
                            t_ms,

                        "path": [],

                        "world_history":
                            deque(
                                maxlen=20
                            ),

                        "speed_history":
                            deque(
                                maxlen=
                                SPEED_SMOOTHING_FRAMES
                            ),

                        "speed_mps":
                            0.0,

                        "speed_kmh":
                            0.0,

                        "vx":
                            0.0,

                        "vy":
                            0.0,

                        # Event detection state
                        "metrics":
                            deque(
                                maxlen=60
                            ),

                        "wrong_way_since":
                            None,

                        "fired": {
                            "wrong_way": False,
                            "speeding_over": False,
                            "speeding_under": False,
                            "harsh_brake": False,
                            "weave": False,
                            "near_miss": False,
                        },
                    }

                track = tracks[
                    track_id
                ]

                track[
                    "last_t_ms"
                ] = t_ms

                # ------------------------------------------------
                # PIXEL PATH
                # ------------------------------------------------

                # path entries are [t_ms, x, y, lane] -- lane folded in here
                # rather than a parallel structure, since path_json is
                # already the array column this data belongs in.
                track[
                    "path"
                ].append(

                    [
                        t_ms,
                        int(x),
                        int(y),
                        lane_number
                    ]

                )

                # ------------------------------------------------
                # WORLD PATH
                # ------------------------------------------------

                if (

                    X is not None

                    and

                    Y is not None

                    and

                    is_plausible_world_step(
                        track["world_history"],
                        timestamp_s,
                        X,
                        Y
                    )

                ):

                    track[
                        "world_history"
                    ].append(

                        [
                            timestamp_s,
                            X,
                            Y
                        ]

                    )

                    # ------------------------------------------------
                    # VELOCITY
                    # ------------------------------------------------

                    (

                        speed_mps,
                        speed_kmh,
                        vx,
                        vy

                    ) = update_speed(

                        track,
                        timestamp_s
                    )

                    # ------------------------------------------------
                    # EVENT DETECTION
                    # ------------------------------------------------

                    # vx/vy/speed_kmh come from numpy (polyfit/mean) --
                    # cast to plain floats once here so every downstream
                    # consumer (meta_json fields) is JSON-safe for free.
                    track["metrics"].append(
                        (timestamp_s, float(vx), float(vy), float(speed_kmh), lane_number)
                    )

                    check_wrong_way(
                        track, track_id, timestamp_s, t_ms, vx, vy, lane_number, incidents
                    )
                    check_speeding(
                        track, track_id, timestamp_s, t_ms, lane_number, incidents
                    )
                    check_harsh_brake(
                        track, track_id, timestamp_s, t_ms, vy, incidents
                    )
                    check_weave(
                        track, track_id, t_ms, incidents
                    )

                    frame_positions[track_id] = (X, Y)

                else:

                    # Missing coordinates or a rejected implausible jump --
                    # hold the track's last known smoothed values (0.0 for
                    # a brand new track) rather than flashing 0 on-screen.
                    speed_mps = track["speed_mps"]

                    speed_kmh = track["speed_kmh"]

                    vx = track["vx"]

                    vy = track["vy"]

                # ====================================================
                # DRAW BOUNDING BOX
                # ====================================================

                cv2.rectangle(

                    frame,

                    (
                        int(x1),
                        int(y1)
                    ),

                    (
                        int(x2),
                        int(y2)
                    ),

                    (0, 255, 0),

                    2
                )

                # ====================================================
                # DRAW CONTACT POINT
                # ====================================================

                cv2.circle(

                    frame,

                    (
                        x,
                        y
                    ),

                    5,

                    (0, 0, 255),

                    -1
                )

                # ====================================================
                # DRAW LABEL
                # ====================================================

                label = (

                    f"ID {track_id} "

                    f"{object_class} "

                    f"{speed_kmh:.1f} km/h"
                )

                cv2.putText(

                    frame,

                    label,

                    (
                        int(x1),
                        max(
                            20,
                            int(y1) - 8
                        )
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.55,

                    (0, 255, 0),

                    2
                )

                # ====================================================
                # WORLD COORDINATES
                # ====================================================

                if X is not None:

                    world_text = (

                        f"X:{X:.1f}m "

                        f"Y:{Y:.1f}m"
                    )

                    cv2.putText(

                        frame,

                        world_text,

                        (
                            x + 10,
                            y + 20
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.45,

                        (255, 255, 0),

                        1
                    )

                # ====================================================
                # LANE NUMBER
                # ====================================================

                lane_text = (
                    f"Lane:{lane_number}"
                    if lane_number is not None
                    else "Lane:-"
                )

                cv2.putText(

                    frame,

                    lane_text,

                    (
                        x + 10,
                        y + 35
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.45,

                    (0, 200, 255),

                    1
                )

                # ====================================================
                # EVENT OVERLAY
                # ====================================================

                event_labels = {
                    "wrong_way": "WRONG WAY",
                    "speeding_over": "SPEEDING+",
                    "speeding_under": "SPEEDING-",
                    "harsh_brake": "HARSH BRAKE",
                    "weave": "WEAVE",
                    "near_miss": "NEAR MISS",
                }

                active_events = [
                    label
                    for key, label in event_labels.items()
                    if track["fired"].get(key)
                ]

                if active_events:

                    cv2.putText(

                        frame,

                        " | ".join(active_events),

                        (
                            x + 10,
                            y + 50
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.5,

                        (0, 0, 255),

                        2
                    )

        check_near_miss(frame_positions, tracks, t_ms, incidents, near_miss_streaks)

        # ====================================================
        # GLOBAL DISPLAY
        # ====================================================

        cv2.putText(

            frame,

            "YOLO + ByteTrack",

            (20, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )

        if CAMERA_HEIGHT_M is not None:
            cv2.putText(

                frame,

                f"Camera height assumption: "
                f"{CAMERA_HEIGHT_M:.1f} m",

                (20, 60),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                (255, 255, 255),

                1
            )

        out.write(frame)

        frame_index += 1

        # --------------------------------------------------------
        # PROGRESS
        # --------------------------------------------------------

        if (
            frame_index
            %
            max(
                1,
                int(fps * 5)
            )
            == 0
        ):

            print(

                f"Processed "
                f"{timestamp_s:.1f}s | "

                f"Tracks: "
                f"{len(tracks)}"

            )

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    out.release()

    print(
        "\n================================"
    )

    print(
        "Processing complete."
    )

    print(
        f"Total tracks: "
        f"{len(tracks)}"
    )

    print(
        f"Output: "
        f"{OUTPUT_VIDEO}"
    )

    print(
        "================================"
    )

    incidents_path = os.path.join(os.path.dirname(VIDEO_PATH), "incidents.json")
    with open(incidents_path, "w", encoding="utf-8") as handle:
        json.dump(incidents, handle, indent=2, default=str)
    print(f"Incidents: {len(incidents)} → {incidents_path}")

    # ========================================================
    # DATABASE
    # ========================================================

    try:

        client = get_client()

        place_id = PLACE_ID
        source_id = SOURCE_ID

        if place_id is None:
            place_id = create_place(client)
            source_id = create_input(client, place_id)
        else:
            client.table("places").update(
                {
                    "ref_length_m": REF_LENGTH_M,
                    "ref_points_json": REF_POINTS,
                    "lanes_json": LANES,
                    "legal_heading": LEGAL_HEADING,
                    "stop_line_json": STOP_LINE,
                }
            ).eq("place_id", place_id).execute()
            _ensure_cleaned_input(client, source_id, place_id)

        keep_ids = {int(inc["track_id"]) for inc in incidents}

        # Re-runs: incidents reference tracks, so wipe children first.
        client.table("incidents").delete().eq("source_id", source_id).execute()
        client.table("tracked_objects").delete().eq("source_id", source_id).execute()

        saved_ids = save_tracks(
            client,
            tracks,
            source_id,
            place_id,
            keep_ids=keep_ids,
        )

        inserted = save_incidents(
            client,
            incidents,
            source_id,
            place_id,
            saved_ids,
        )

        print(f"place_id: {place_id}")
        print(f"source_id: {source_id}")
        print(f"incidents local: {len(incidents)}  db: {inserted}")

        ingest_path = os.path.join(os.path.dirname(VIDEO_PATH), "ingest.json")
        if os.path.isfile(ingest_path):
            with open(ingest_path, encoding="utf-8") as handle:
                rec = json.load(handle)
            rec["place_id"] = place_id
            rec["source_id"] = source_id
            with open(ingest_path, "w", encoding="utf-8") as handle:
                json.dump(rec, handle, indent=2)

    except Exception as e:

        print(
            "\nDatabase upload failed:"
        )

        print(e)

        print(
            "Video processing was still successful."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()