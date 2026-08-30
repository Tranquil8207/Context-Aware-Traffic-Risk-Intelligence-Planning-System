"""Click-to-box helper for manual lane polygons, for objectdetection.py.

Pauses on a frame of the target video and lets you draw as many lane
polygons as you want, each with as many points as you want (so curves and
merges don't need special-case code). After each lane's boundary, you then
click a 2-point arrow marking that lane's legal direction of travel -- this
gets converted through the existing homography (from calibration.json) into
a world-space heading in degrees, so it's directly comparable to a live
vehicle's computed heading (atan2(vx, vy), same convention: 0 = straight up
the road along +Y, increasing toward +X).

Requires calibration.json's homography section to already exist (run
calibrate_homography.py first) -- pixel-space angles are badly distorted by
perspective, so heading is only meaningful once converted to world space.
Without it, lanes still save fine, just with "heading": null.

Saves into calibration.json's "lanes" section. objectdetection.py loads
that file automatically on startup - no copy-pasting values between files.

Controls:
    left-click     add a point to the lane boundary currently being drawn,
                   or (once a boundary is finished) add a point to its
                   direction arrow (start, then end -- 2 clicks)
    u              undo the last point (boundary or arrow, whichever is
                   in progress)
    Enter / c      finish the current lane boundary (needs >=3 points); or,
                   while placing its direction arrow, skip the heading for
                   this lane and finalize it with heading = null
    z              undo the last *finished* lane entirely
    r              reset the current in-progress boundary or arrow
    n              next frame
    p              previous frame
    q / Esc        finish (auto-closes an in-progress boundary with >=3
                   points, dropping any incomplete arrow), then save results

Lane numbers are assigned in the order you finish each lane, starting at 1
-- box lanes left-to-right (or whatever order matters to you) so the
numbers come out meaningful.

Usage:
    python calibrate_lanes.py [video_path] [start_frame]
"""

import math
import sys

import cv2
import numpy as np

from calibration_io import CALIBRATION_PATH, load_calibration, update_calibration

DEFAULT_VIDEO_PATH = "YTDown.com_YouTube_Indian-Traffic-Vehicles-Highway-Footage-_Media_tQnVX3nj3Co_002_720p (1).mp4"

LANE_COLORS = [
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
]
PENDING_COLOR = (0, 200, 200)
ARROW_COLOR = (255, 255, 255)

# finished lane entries: {"polygon": [(x,y), ...], "heading": float | None,
# "_arrow_px": ((x,y), (x,y)) | None} -- "_arrow_px" is display-only, the
# clicked pixel arrow, stripped before saving.
finished_lanes: list[dict] = []
current_polygon: list[tuple[int, int]] = []
pending_polygon: list[tuple[int, int]] | None = None
current_arrow: list[tuple[int, int]] = []

HOMOGRAPHY: np.ndarray | None = None


def build_homography(homography_cal: dict) -> np.ndarray | None:
    src = homography_cal.get("src")
    lane_width_m = homography_cal.get("lane_width_m")
    reference_distance_m = homography_cal.get("reference_distance_m")

    if not src or lane_width_m is None or reference_distance_m is None:
        return None

    src_pts = np.float32(src)
    dst_pts = np.float32([
        [0.0, 0.0],
        [lane_width_m, 0.0],
        [0.0, reference_distance_m],
        [lane_width_m, reference_distance_m],
    ])

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def pixel_to_world(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    point = np.array([[[float(x), float(y)]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, H)
    return float(transformed[0][0][0]), float(transformed[0][0][1])


def compute_heading_degrees(H: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> float:
    x1, y1 = pixel_to_world(H, *start)
    x2, y2 = pixel_to_world(H, *end)
    return math.degrees(math.atan2(x2 - x1, y2 - y1))


def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if pending_polygon is not None:
        if len(current_arrow) < 2:
            current_arrow.append((x, y))
        if len(current_arrow) == 2:
            finalize_pending_lane()
    else:
        current_polygon.append((x, y))


def lane_color(index: int) -> tuple[int, int, int]:
    return LANE_COLORS[index % len(LANE_COLORS)]


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    display = frame.copy()

    for i, entry in enumerate(finished_lanes):
        color = lane_color(i)
        pts = np.array(entry["polygon"], dtype=np.int32)
        cv2.polylines(display, [pts], True, color, 2)
        centroid = pts.mean(axis=0).astype(int)
        heading = entry["heading"]
        label = f"Lane {i + 1}" + (f"  {heading:.0f} deg" if heading is not None else "  no heading")
        cv2.putText(
            display, label, (int(centroid[0]), int(centroid[1])),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        if entry["_arrow_px"] is not None:
            start, end = entry["_arrow_px"]
            cv2.arrowedLine(display, start, end, color, 2, tipLength=0.3)

    if current_polygon:
        color = lane_color(len(finished_lanes))
        for point in current_polygon:
            cv2.circle(display, point, 4, color, -1)
        if len(current_polygon) > 1:
            cv2.polylines(display, [np.array(current_polygon, dtype=np.int32)], False, color, 2)

    if pending_polygon is not None:
        color = PENDING_COLOR
        cv2.polylines(display, [np.array(pending_polygon, dtype=np.int32)], True, color, 2)
        for point in current_arrow:
            cv2.circle(display, point, 5, ARROW_COLOR, -1)
        if len(current_arrow) == 1:
            cv2.line(display, current_arrow[0], current_arrow[0], ARROW_COLOR, 2)

    status = f"Lanes finished: {len(finished_lanes)}   "
    if pending_polygon is not None:
        status += f"Click direction arrow: {len(current_arrow)}/2 points"
    else:
        status += f"Current lane points: {len(current_polygon)}"
    cv2.putText(display, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(
        display,
        "click=add point  u=undo point  enter/c=finish/skip-heading  z=undo lane",
        (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    cv2.putText(
        display,
        "r=reset current  n/p=next/prev frame  q=quit+save",
        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
    )
    if HOMOGRAPHY is None:
        cv2.putText(
            display,
            "No homography in calibration.json -- headings will be saved as null",
            (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )

    return display


def load_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(index, 0))
    ok, frame = cap.read()
    return frame if ok else None


def finish_current_lane() -> None:
    global pending_polygon

    if len(current_polygon) < 3:
        print(f"\nNeed at least 3 points to finish a lane, have {len(current_polygon)} -- ignored.")
        return

    if HOMOGRAPHY is None:
        finished_lanes.append({"polygon": list(current_polygon), "heading": None, "_arrow_px": None})
        print(f"\nLane {len(finished_lanes)} finished with {len(current_polygon)} points (no heading).")
        current_polygon.clear()
        return

    pending_polygon = list(current_polygon)
    current_polygon.clear()
    print(
        f"\nLane boundary finished with {len(pending_polygon)} points. "
        "Click 2 points for its direction of travel (start, then end), "
        "or press Enter/'c' to skip the heading for this lane."
    )


def finalize_pending_lane() -> None:
    global pending_polygon

    heading = None
    if len(current_arrow) == 2:
        heading = compute_heading_degrees(HOMOGRAPHY, current_arrow[0], current_arrow[1])

    finished_lanes.append({
        "polygon": pending_polygon,
        "heading": heading,
        "_arrow_px": (current_arrow[0], current_arrow[1]) if len(current_arrow) == 2 else None,
    })

    heading_text = f"{heading:.1f} deg" if heading is not None else "no heading"
    print(f"Lane {len(finished_lanes)} saved with {len(finished_lanes[-1]['polygon'])} points, heading: {heading_text}")

    pending_polygon = None
    current_arrow.clear()


def save_result() -> None:
    if not finished_lanes:
        print("\nNo lanes finished -- nothing saved.")
        return

    lanes_data = [
        {
            "lane": i + 1,
            "polygon": [[x, y] for x, y in entry["polygon"]],
            "heading": entry["heading"],
        }
        for i, entry in enumerate(finished_lanes)
    ]
    update_calibration("lanes", lanes_data)

    print(f"\nSaved {len(lanes_data)} lane(s) to {CALIBRATION_PATH}.")
    print(
        "objectdetection.py will pick this up automatically on its next "
        "run - no copy-pasting needed."
    )


def main() -> None:
    global HOMOGRAPHY

    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    frame_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    HOMOGRAPHY = build_homography(load_calibration().get("homography", {}))
    if HOMOGRAPHY is None:
        print(
            "No homography calibration found (or incomplete) in "
            f"{CALIBRATION_PATH}. Run calibrate_homography.py first if you "
            "want per-lane legal_heading; lanes will still save fine "
            "otherwise, just with heading: null.\n"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame = load_frame(cap, frame_index)
    if frame is None:
        raise RuntimeError(f"Could not read starting frame {frame_index} from {video_path}")

    window = "Lane boxing - click points, Enter to finish a lane"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print("Click points to trace the first lane's boundary (as many as you want,")
    print("so curves are fine), then press Enter or 'c' to finish it. If a")
    print("homography is loaded, you'll then click a 2-point direction-of-travel")
    print("arrow for that lane before moving on to the next one.\n")

    while True:
        cv2.imshow(window, draw_overlay(frame))
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            break
        elif key in (13, ord("c")):  # Enter or c
            if pending_polygon is not None:
                finalize_pending_lane()
            else:
                finish_current_lane()
        elif key == ord("u"):
            if pending_polygon is not None:
                if current_arrow:
                    current_arrow.pop()
            elif current_polygon:
                current_polygon.pop()
        elif key == ord("z"):
            if finished_lanes:
                removed = finished_lanes.pop()
                print(f"\nRemoved lane with {len(removed['polygon'])} points. {len(finished_lanes)} lane(s) remain.")
        elif key == ord("r"):
            if pending_polygon is not None:
                current_arrow.clear()
                print("\nDirection arrow cleared.")
            else:
                current_polygon.clear()
                print("\nCurrent lane points cleared.")
        elif key == ord("n"):
            next_frame = load_frame(cap, frame_index + 1)
            if next_frame is not None:
                frame_index += 1
                frame = next_frame
            else:
                print("End of video - can't advance further.")
        elif key == ord("p") and frame_index > 0:
            prev_frame = load_frame(cap, frame_index - 1)
            if prev_frame is not None:
                frame_index -= 1
                frame = prev_frame

    if pending_polygon is not None:
        print("\nQuitting with an incomplete direction arrow -- saving that lane with no heading.")
        current_arrow.clear()
        finalize_pending_lane()
    elif len(current_polygon) >= 3:
        print("\nAuto-finishing the in-progress lane before quitting.")
        finish_current_lane()
        if pending_polygon is not None:
            finalize_pending_lane()
    elif current_polygon:
        print(f"\nDropping in-progress lane with only {len(current_polygon)} point(s) (need >=3).")

    cap.release()
    cv2.destroyAllWindows()
    save_result()


if __name__ == "__main__":
    main()
