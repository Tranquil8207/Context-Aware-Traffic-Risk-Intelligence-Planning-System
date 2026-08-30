import math
import os
import sys
from collections import deque

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from supabase import Client
from ultralytics import YOLO

from calibration_io import load_calibration
from supabase_client import get_client

# The YOLOP repo has no top-level `YOLOP` package to import from — its
# importable code lives under lib/. Add the cloned repo folder to sys.path
# so `lib` resolves, matching the repo's own tools/demo.py.
YOLOP_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "YOLOP")
if YOLOP_ROOT not in sys.path:
    sys.path.append(YOLOP_ROOT)

from lib.config import cfg  # noqa: E402
from lib.models import get_net  # noqa: E402


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "istockphoto-1282097660-640_adpp_is.mp4"

# Your existing object detector
OBJECT_MODEL_PATH = "yolo26m.pt"

# YOLOP lane model
# Download the official YOLOP End-to-end.pth
LANE_MODEL_PATH = "YOLOP/weights/End-to-end.pth"

OUTPUT_VIDEO = "stage1_lane_tracking.mp4"

CONFIDENCE = 0.35

# ------------------------------------------------------------
# ROAD / CAMERA ASSUMPTIONS
# ------------------------------------------------------------
#
# Loaded from calibration.json if it exists (written by
# calibrate_homography.py / calibrate_lanes.py), so recalibrating never
# requires editing this file. The literal values below are the fallback
# used only if calibration.json is missing or incomplete -- currently the
# last known-good calibration from this project's own history.

_CALIBRATION = load_calibration()
_HOMOGRAPHY_CAL = _CALIBRATION.get("homography", {})

HOMOGRAPHY_SRC = _HOMOGRAPHY_CAL.get("src", [
    [142, 431],   # near-left
    [697, 431],   # near-right
    [294, 163],   # far-left
    [438, 163],   # far-right
])

LANE_WIDTH_M = _HOMOGRAPHY_CAL.get("lane_width_m", 10.5)

# LANE_WIDTH_M is the combined width of NUM_LANES equal-width lanes, used to
# calibrate the homography. Each individual lane's width for lane-number
# classification is LANE_WIDTH_M / NUM_LANES. Only valid if the lanes are
# uniform and parallel across the calibrated stretch of road.
NUM_LANES = 3
SINGLE_LANE_WIDTH_M = LANE_WIDTH_M / NUM_LANES

# Engineering assumption for high-mounted expressway CCTV.
# NOT a verified DME camera specification.
CAMERA_HEIGHT_M = 8.0

# Used for homography calibration.
REFERENCE_DISTANCE_M = _HOMOGRAPHY_CAL.get("reference_distance_m", 120.0)

SPEED_SMOOTHING_FRAMES = 5

# Reject physically impossible instantaneous speeds
MAX_REASONABLE_SPEED_KMH = 180.0

# ------------------------------------------------------------
# Road metadata
# ------------------------------------------------------------

LAT = None
LNG = None

ROAD_KIND = "expressway"

SPEED_LIMIT_KMH = None

REF_LENGTH_M = SINGLE_LANE_WIDTH_M

# The pixel points actually used above, whether from calibration.json or
# the fallback -- this is what places.ref_points_json records.
REF_POINTS = HOMOGRAPHY_SRC

STOP_LINE = None
LEGAL_HEADING = None

# Loaded from calibration.json's "lanes" section if present (written by
# calibrate_lanes.py); falls back to the last known-good boxing otherwise.
LANES = _CALIBRATION.get("lanes", [
    {"lane": 1, "polygon": [[150, 431], [334, 431], [346, 126], [314, 127], [150, 430]]},
    {"lane": 2, "polygon": [[338, 431], [350, 127], [376, 127], [516, 431], [338, 431]]},
    {"lane": 3, "polygon": [[378, 127], [521, 430], [694, 431], [404, 127], [380, 127]]},
])

HAS_SIGNAL = 0
CITY_PRIOR = 1.0
IS_NIGHT = 0
IS_RAIN = 0
IS_RUSH = 0


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
# YOLOP LANE MODEL
# ============================================================

class YOLOPLaneDetector:

    def __init__(self, weights_path, device="cpu"):

        self.device = torch.device(device)

        print("\nLoading YOLOP lane model...")

        self.model = get_net(cfg)

        checkpoint = torch.load(
            weights_path,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("YOLOP loaded.")

    def preprocess(self, frame):

        # YOLOP expects 640x640
        img = cv2.resize(
            frame,
            (640, 640)
        )

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        tensor = self.transform(img_rgb)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        return tensor

    def detect_lane_mask(self, frame):

        original_h, original_w = frame.shape[:2]

        tensor = self.preprocess(frame)

        with torch.no_grad():

            _, _, lane_output = self.model(
                tensor
            )

        # YOLOP lane segmentation
        lane_output = torch.nn.functional.interpolate(
            lane_output,
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False
        )

        lane_mask = torch.argmax(
            lane_output,
            dim=1
        )

        lane_mask = (
            lane_mask
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )

        return lane_mask


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

    HOMOGRAPHY_SRC / LANE_WIDTH_M / REFERENCE_DISTANCE_M come from
    calibration.json (see the CONFIGURATION section) -- re-run
    calibrate_homography.py if the camera or video ever changes, rather
    than editing values here.
    """

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
# LANES is manually boxed per video via calibrate_lanes.py, as a list of
# {"lane": n, "polygon": [[x, y], ...]} pixel-space polygons -- one per
# lane, drawn by hand so curves/merges don't need special-case code.
# LANE_POLYGONS caches the parsed numpy polygons; built once in main().

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
# LANE PROCESSING
# ============================================================

def clean_lane_mask(mask):

    binary = np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel
    )

    return binary


def draw_lane_mask(frame, mask):

    overlay = frame.copy()

    lane_pixels = mask > 0

    # Green lane visualization
    overlay[lane_pixels] = (
        0.5 * overlay[lane_pixels] +
        0.5 * np.array(
            [0, 255, 255]
        )
    ).astype(np.uint8)

    return cv2.addWeighted(
        frame,
        0.7,
        overlay,
        0.3,
        0
    )


# ============================================================
# LANE CURVE EXTRACTION
# ============================================================

def extract_lane_points(mask):

    """
    Extract center points of detected lane markings
    row-by-row.

    Returns:
        [(x, y), ...]
    """

    points = []

    h, w = mask.shape

    # Focus on road region
    start_y = int(h * 0.40)

    for y in range(
        start_y,
        h,
        5
    ):

        xs = np.where(
            mask[y] > 0
        )[0]

        if len(xs) == 0:
            continue

        # Group contiguous lane pixels
        groups = []

        current = [
            xs[0]
        ]

        for i in range(
            1,
            len(xs)
        ):

            if xs[i] - xs[i - 1] <= 3:
                current.append(
                    xs[i]
                )
            else:

                if len(current) >= 2:
                    groups.append(
                        current
                    )

                current = [
                    xs[i]
                ]

        if len(current) >= 2:
            groups.append(
                current
            )

        for group in groups:

            x_center = int(
                np.mean(group)
            )

            points.append(
                (x_center, y)
            )

    return points


def draw_lane_points(
    frame,
    lane_points
):

    for x, y in lane_points:

        cv2.circle(
            frame,
            (x, y),
            2,
            (255, 255, 0),
            -1
        )

    return frame


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

MAX_REASONABLE_SPEED_MPS = MAX_REASONABLE_SPEED_KMH / 3.6


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

    source_id = os.path.basename(
        VIDEO_PATH
    )

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
            None,

        "clock_note":
            "demo-time",

        "notes":
            "Delhi-Meerut Expressway traffic video",
    }

    (
        client
        .table("cleaned_inputs")
        .insert(row)
        .execute()
    )

    return source_id


def save_tracks(
    client,
    tracks,
    source_id,
    place_id
):

    rows = []

    for track_id, data in tracks.items():

        duration_s = (
            data["last_t_ms"]
            -
            data["first_t_ms"]
        ) / 1000.0

        if duration_s < 0.4:
            continue

        rows.append({

            "track_id":
                int(track_id),

            "source_id":
                source_id,

            "place_id":
                place_id,

            "class":
                data["class"],

            "path_json":
                data["path"],

            "mean_speed_kmh":
                float(
                    data["speed_kmh"]
                ),

            "object_risk":
                None,
        })

    if rows:

        (
            client
            .table("tracked_objects")
            .insert(rows)
            .execute()
        )


# ============================================================
# MAIN
# ============================================================

def main():

    global HOMOGRAPHY
    global LANE_POLYGONS

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
            "Run calibrate_lanes.py to save lane polygons into "
            "calibration.json."
        )

    # --------------------------------------------------------
    # YOLOP
    # --------------------------------------------------------

    if not os.path.exists(
        LANE_MODEL_PATH
    ):

        raise FileNotFoundError(
            "\nYOLOP weights not found:\n"
            f"{LANE_MODEL_PATH}\n\n"
            "Download the official End-to-end.pth "
            "weights from the YOLOP repository."
        )

    lane_model = YOLOPLaneDetector(
        LANE_MODEL_PATH,
        device="cuda"
        if torch.cuda.is_available()
        else "cpu"
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

        # ====================================================
        # LANE SEGMENTATION
        # ====================================================

        lane_mask = (
            lane_model
            .detect_lane_mask(
                frame
            )
        )

        lane_mask = clean_lane_mask(
            lane_mask
        )

        lane_points = (
            extract_lane_points(
                lane_mask
            )
        )

        # Draw lane segmentation
        frame = draw_lane_mask(
            frame,
            lane_mask
        )

        frame = draw_lane_points(
            frame,
            lane_points
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
        # GLOBAL DISPLAY
        # ====================================================

        cv2.putText(

            frame,

            "YOLO + ByteTrack + YOLOP",

            (20, 30),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )

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

    # ========================================================
    # DATABASE
    # ========================================================

    try:

        client = get_client()

        place_id = create_place(
            client
        )

        source_id = create_input(
            client,
            place_id
        )

        save_tracks(

            client,

            tracks,

            source_id,

            place_id
        )

        print(
            f"place_id: {place_id}"
        )

        print(
            f"source_id: {source_id}"
        )

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