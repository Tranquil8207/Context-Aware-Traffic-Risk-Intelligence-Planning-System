import os
import cv2
from supabase import Client
from ultralytics import YOLO

from supabase_client import get_client


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "istockphoto-1282097660-640_adpp_is.mp4"
MODEL_PATH = "yolo26m.pt"

OUTPUT_VIDEO = "stage1_tracking.mp4"

CONFIDENCE = 0.35
MIN_TRACK_TIME_S = 0.4

# Junction metadata
LAT = None
LNG = None
ROAD_KIND = "expressway"
SPEED_LIMIT_KMH = None

# Three lanes, each 3.5 m wide.
# Pixel geometry is deliberately left empty until manually defined.
REF_LENGTH_M = 3.5
REF_POINTS = None
STOP_LINE = None
LEGAL_HEADING = None
LANES = None

HAS_SIGNAL = 0
CITY_PRIOR = 1.0
IS_NIGHT = 0
IS_RAIN = 0
IS_RUSH = 0

# YOLO COCO classes
CLASS_MAP = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ============================================================
# DATABASE
# ============================================================

def create_place(client: Client) -> int:
    # Supabase encodes plain dicts/lists to jsonb columns itself; no
    # json.dumps or ::jsonb cast needed.
    row = {
        "lat": LAT,
        "lng": LNG,
        "road_kind": ROAD_KIND,
        "speed_limit_kmh": SPEED_LIMIT_KMH,
        "ref_length_m": REF_LENGTH_M,
        "ref_points_json": REF_POINTS,
        "stop_line_json": STOP_LINE,
        "legal_heading": LEGAL_HEADING,
        "lanes_json": LANES,
        "has_signal": HAS_SIGNAL,
        "city_prior": CITY_PRIOR,
        "is_night": IS_NIGHT,
        "is_rain": IS_RAIN,
        "is_rush": IS_RUSH,
    }

    response = client.table("places").insert(row).execute()
    return response.data[0]["place_id"]


def create_input(client: Client, place_id: int) -> str:
    source_id = os.path.basename(VIDEO_PATH)

    row = {
        "source_id": source_id,
        "kind": "video",
        "place_id": place_id,
        "file_name": os.path.basename(VIDEO_PATH),
        "clock_start": None,
        "clock_note": "demo-time",
        "notes": "Delhi-Meerut Expressway traffic video",
    }

    client.table("cleaned_inputs").insert(row).execute()
    return source_id


def save_tracks(client: Client, tracks: dict, source_id: str, place_id: int) -> None:
    rows = []

    for track_id, data in tracks.items():
        duration_s = (data["last_t_ms"] - data["first_t_ms"]) / 1000.0

        if duration_s < MIN_TRACK_TIME_S:
            continue

        rows.append({
            # track_id/x/y arrive as numpy ints from YOLO; cast to plain
            # Python int so the request body can be JSON-encoded.
            "track_id": int(track_id),
            "source_id": source_id,
            "place_id": place_id,
            "class": data["class"],
            "path_json": data["path"],
            "mean_speed_kmh": None,
            "object_risk": None,
        })

    if rows:
        client.table("tracked_objects").insert(rows).execute()


# ============================================================
# VIDEO / TRACKING
# ============================================================

def main():
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(
            f"Video not found: {VIDEO_PATH}"
        )

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {VIDEO_PATH}"
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Video: {width} x {height}")
    print(f"FPS: {fps:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (width, height),
    )

    if not out.isOpened():
        raise RuntimeError(
            "Could not create output video."
        )

    tracks = {}
    frame_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        t_ms = int((frame_index / fps) * 1000)

        results = model.track(
            frame,
            persist=True,
            conf=CONFIDENCE,
            classes=[0, 2, 3, 5, 7],
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:

            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            ids = result.boxes.id.cpu().numpy().astype(int)

            for bbox, cls, conf, track_id in zip(
                boxes,
                classes,
                confidences,
                ids,
            ):

                if float(conf) < CONFIDENCE:
                    continue

                class_id = int(cls)

                if class_id not in CLASS_MAP:
                    object_class = "other"
                else:
                    object_class = CLASS_MAP[class_id]

                x1, y1, x2, y2 = bbox

                # REQUIRED BY STAGE 1:
                # bottom-centre of bounding box
                x = int((x1 + x2) / 2)
                y = int(y2)

                if track_id not in tracks:
                    tracks[track_id] = {
                        "class": object_class,
                        "first_t_ms": t_ms,
                        "last_t_ms": t_ms,
                        "path": [],
                    }

                track = tracks[track_id]

                track["last_t_ms"] = t_ms

                track["path"].append(
                    [t_ms, x, y]
                )

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (0, 255, 0),
                    2,
                )

                # Draw ID
                label = (
                    f"ID {track_id} "
                    f"{object_class} "
                    f"{conf:.2f}"
                )

                cv2.putText(
                    frame,
                    label,
                    (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                # Draw bottom-centre point
                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 0, 255),
                    -1,
                )

        out.write(frame)

        frame_index += 1

        if frame_index % int(fps * 5) == 0:
            elapsed = frame_index / fps
            print(
                f"Processed {elapsed:.1f}s | "
                f"active IDs: {len(tracks)}"
            )

    cap.release()
    out.release()

    print("\nTracking complete.")
    print(f"Total raw IDs: {len(tracks)}")

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    client = get_client()

    place_id = create_place(client)

    source_id = create_input(
        client,
        place_id,
    )

    save_tracks(
        client,
        tracks,
        source_id,
        place_id,
    )

    print(f"place_id: {place_id}")
    print(f"source_id: {source_id}")

    # --------------------------------------------------------
    # TRACK SUMMARY
    # --------------------------------------------------------

    valid_tracks = 0

    for track_id, data in tracks.items():

        duration_s = (
            data["last_t_ms"] -
            data["first_t_ms"]
        ) / 1000.0

        if duration_s >= MIN_TRACK_TIME_S:
            valid_tracks += 1

    print(
        f"Valid tracks (>={MIN_TRACK_TIME_S}s): "
        f"{valid_tracks}"
    )

    print(
        f"\nOutput video: {OUTPUT_VIDEO}"
    )


if __name__ == "__main__":
    main()