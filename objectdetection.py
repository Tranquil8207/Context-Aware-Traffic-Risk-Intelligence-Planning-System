import os
import json
import cv2
import psycopg2
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "istockphoto-1282097660-640_adpp_is.mp4"
MODEL_PATH = "yolo26m.pt"

OUTPUT_VIDEO = "stage1_tracking.mp4"

CONFIDENCE = 0.35
MIN_TRACK_TIME_S = 0.4

# PostgreSQL connection
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "traffic")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

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

def connect_db():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def create_place(conn):
    query = """
        INSERT INTO places (
            lat,
            lng,
            road_kind,
            speed_limit_kmh,
            ref_length_m,
            ref_points_json,
            stop_line_json,
            legal_heading,
            lanes_json,
            has_signal,
            city_prior,
            is_night,
            is_rain,
            is_rush
        )
        VALUES (
            %s, %s, %s, %s, %s,
            %s::jsonb,
            %s::jsonb,
            %s,
            %s::jsonb,
            %s, %s, %s, %s, %s
        )
        RETURNING place_id;
    """

    values = (
        LAT,
        LNG,
        ROAD_KIND,
        SPEED_LIMIT_KMH,
        REF_LENGTH_M,
        json.dumps(REF_POINTS) if REF_POINTS is not None else None,
        json.dumps(STOP_LINE) if STOP_LINE is not None else None,
        LEGAL_HEADING,
        json.dumps(LANES) if LANES is not None else None,
        HAS_SIGNAL,
        CITY_PRIOR,
        IS_NIGHT,
        IS_RAIN,
        IS_RUSH,
    )

    with conn.cursor() as cur:
        cur.execute(query, values)
        place_id = cur.fetchone()[0]

    conn.commit()
    return place_id


def create_input(conn, place_id):
    query = """
        INSERT INTO cleaned_inputs (
            source_id,
            kind,
            place_id,
            file_name,
            clock_start,
            clock_note,
            notes
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s);
    """

    source_id = os.path.basename(VIDEO_PATH)

    values = (
        source_id,
        "video",
        place_id,
        os.path.basename(VIDEO_PATH),
        None,
        "demo-time",
        "Delhi-Meerut Expressway traffic video",
    )

    with conn.cursor() as cur:
        cur.execute(query, values)

    conn.commit()
    return source_id


def save_tracks(conn, tracks, source_id, place_id):
    query = """
        INSERT INTO tracked_objects (
            track_id,
            source_id,
            place_id,
            class,
            path_json,
            plate_text,
            mean_speed_kmh,
            object_risk
        )
        VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s, %s);
    """

    with conn.cursor() as cur:
        for track_id, data in tracks.items():

            duration_s = (
                data["last_t_ms"] - data["first_t_ms"]
            ) / 1000.0

            if duration_s < MIN_TRACK_TIME_S:
                continue

            cur.execute(
                query,
                (
                    track_id,
                    source_id,
                    place_id,
                    data["class"],
                    json.dumps(data["path"]),
                    "",
                    None,
                    None,
                ),
            )

    conn.commit()


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

    if width != 1920 or height != 1080:
        raise RuntimeError(
            f"Expected 1920x1080, got {width}x{height}"
        )

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

    conn = connect_db()

    try:
        place_id = create_place(conn)

        source_id = create_input(
            conn,
            place_id,
        )

        save_tracks(
            conn,
            tracks,
            source_id,
            place_id,
        )

        print(f"place_id: {place_id}")
        print(f"source_id: {source_id}")

    finally:
        conn.close()

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