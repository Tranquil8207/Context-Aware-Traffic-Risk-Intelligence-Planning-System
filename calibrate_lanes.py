"""Click-to-box helper for manual lane polygons, for objectdetection.py.

Pauses on a frame of the target video and lets you draw as many lane
polygons as you want, each with as many points as you want (so curves and
merges don't need special-case code) -- then saves the result into
calibration.json's "lanes" section. objectdetection.py loads that file
automatically on startup - no copy-pasting values between files.

Controls:
    left-click     add a point to the lane currently being drawn
    u              undo the last point in the lane currently being drawn
    Enter / c      finish the current lane (needs >=3 points) and start a
                   new one
    z              undo the last *finished* lane entirely
    r              reset the current in-progress lane's points
    n              next frame
    p              previous frame
    q / Esc        finish (auto-closes an in-progress lane with >=3 points),
                   then save results

Lane numbers are assigned in the order you finish each polygon, starting at
1 -- box lanes left-to-right (or whatever order matters to you) so the
numbers come out meaningful.

Usage:
    python calibrate_lanes.py [video_path] [start_frame]
"""

import sys

import cv2
import numpy as np

from calibration_io import CALIBRATION_PATH, update_calibration

DEFAULT_VIDEO_PATH = "istockphoto-1282097660-640_adpp_is.mp4"

LANE_COLORS = [
    (255, 0, 0),
    (0, 165, 255),
    (255, 0, 255),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
]

finished_lanes: list[list[tuple[int, int]]] = []
current_polygon: list[tuple[int, int]] = []


def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        current_polygon.append((x, y))


def lane_color(index: int) -> tuple[int, int, int]:
    return LANE_COLORS[index % len(LANE_COLORS)]


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    display = frame.copy()

    for i, polygon in enumerate(finished_lanes):
        color = lane_color(i)
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(display, [pts], True, color, 2)
        centroid = pts.mean(axis=0).astype(int)
        cv2.putText(
            display,
            f"Lane {i + 1}",
            (int(centroid[0]), int(centroid[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    if current_polygon:
        color = lane_color(len(finished_lanes))
        for point in current_polygon:
            cv2.circle(display, point, 4, color, -1)
        if len(current_polygon) > 1:
            cv2.polylines(display, [np.array(current_polygon, dtype=np.int32)], False, color, 2)

    status = (
        f"Lanes finished: {len(finished_lanes)}   "
        f"Current lane points: {len(current_polygon)}"
    )
    cv2.putText(display, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(
        display,
        "click=add point  u=undo point  enter/c=finish lane  z=undo lane",
        (20, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        display,
        "r=reset current lane  n/p=next/prev frame  q=quit+save",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return display


def load_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(index, 0))
    ok, frame = cap.read()
    return frame if ok else None


def finish_current_lane() -> None:
    if len(current_polygon) < 3:
        print(f"\nNeed at least 3 points to finish a lane, have {len(current_polygon)} -- ignored.")
        return

    finished_lanes.append(list(current_polygon))
    print(f"\nLane {len(finished_lanes)} finished with {len(current_polygon)} points.")
    current_polygon.clear()


def save_result() -> None:
    if not finished_lanes:
        print("\nNo lanes finished -- nothing saved.")
        return

    lanes_data = [
        {"lane": i + 1, "polygon": [[x, y] for x, y in polygon]}
        for i, polygon in enumerate(finished_lanes)
    ]
    update_calibration("lanes", lanes_data)

    print(f"\nSaved {len(lanes_data)} lane(s) to {CALIBRATION_PATH}.")
    print(
        "objectdetection.py will pick this up automatically on its next "
        "run - no copy-pasting needed."
    )


def main() -> None:
    video_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    frame_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

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
    print("so curves are fine), then press Enter or 'c' to finish it and start")
    print("the next lane. Box as many lanes as exist in the frame.\n")

    while True:
        cv2.imshow(window, draw_overlay(frame))
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            break
        elif key in (13, ord("c")):  # Enter or c
            finish_current_lane()
        elif key == ord("u"):
            if current_polygon:
                current_polygon.pop()
        elif key == ord("z"):
            if finished_lanes:
                removed = finished_lanes.pop()
                print(f"\nRemoved lane with {len(removed)} points. {len(finished_lanes)} lane(s) remain.")
        elif key == ord("r"):
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

    if len(current_polygon) >= 3:
        print("\nAuto-finishing the in-progress lane before quitting.")
        finish_current_lane()
    elif current_polygon:
        print(f"\nDropping in-progress lane with only {len(current_polygon)} point(s) (need >=3).")

    cap.release()
    cv2.destroyAllWindows()
    save_result()


if __name__ == "__main__":
    main()
