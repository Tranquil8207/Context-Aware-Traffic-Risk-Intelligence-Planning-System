"""Click-to-calibrate helper for the homography in objectdetection.py.

Pauses on a frame of the target video, lets you click the 4 points that
define a known real-world rectangle on the road, and saves the result into
calibration.json's "homography" section. objectdetection.py loads that file
automatically on startup - no copy-pasting values between files.

Pick two real, identifiable points on the road at a known distance apart
(near-left/near-right - e.g. where a lane-width road marking crosses a stop
line), then the same lane's edges further up the road (far-left/far-right).
The near pair and the far pair should each be the same real lane width
apart, and the far pair should be some known reference distance further down
the road than the near pair. After the 4th point is clicked, you'll be
prompted in the terminal to enter both real-world measurements (this also
feeds objectdetection.py's REF_LENGTH_M, since a homography can't derive
real-world scale from pixels alone - it always needs an asserted real-world
measurement).

Controls:
    left-click   record a point (up to 4, in order: near-left, near-right,
                 far-left, far-right)
    r            reset points on the current frame
    n            next frame
    p            previous frame
    q / Esc      quit, then answer the measurement prompts and save results

Usage:
    python calibrate_homography.py [video_path] [start_frame]
                                    [--lane-width M] [--reference-distance M]

    --lane-width/--reference-distance just pre-fill the terminal prompt's
    default after clicking - they're shown and must be confirmed (or
    overridden) interactively, never used silently.
"""

import argparse

import cv2
import numpy as np

from calibration_io import CALIBRATION_PATH, update_calibration

DEFAULT_VIDEO_PATH = "istockphoto-1282097660-640_adpp_is.mp4"

# Fallback real-world measurements if not overridden on the command line.
# Must match LANE_WIDTH_M / REFERENCE_DISTANCE_M in objectdetection.py.
DEFAULT_LANE_WIDTH_M = 3.5
DEFAULT_REFERENCE_DISTANCE_M = 20.0

POINT_LABELS = ["near-left", "near-right", "far-left", "far-right"]
POINT_COLORS = [(0, 255, 0), (0, 255, 0), (0, 165, 255), (0, 165, 255)]

points: list[tuple[int, int]] = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Click-to-calibrate helper for objectdetection.py's homography."
    )
    parser.add_argument("video_path", nargs="?", default=DEFAULT_VIDEO_PATH)
    parser.add_argument("start_frame", nargs="?", type=int, default=0)
    parser.add_argument(
        "--lane-width",
        type=float,
        default=DEFAULT_LANE_WIDTH_M,
        help=(
            "Default shown in the post-click terminal prompt for the real "
            "lane width in meters (distance between near-left/near-right, "
            "and between far-left/far-right). Still confirmed interactively "
            f"after clicking, never used silently. Default: {DEFAULT_LANE_WIDTH_M}m."
        ),
    )
    parser.add_argument(
        "--reference-distance",
        type=float,
        default=DEFAULT_REFERENCE_DISTANCE_M,
        help=(
            "Default shown in the post-click terminal prompt for the real "
            "distance in meters between the near line and far line. Still "
            "confirmed interactively after clicking, never used silently. "
            f"Default: {DEFAULT_REFERENCE_DISTANCE_M}m."
        ),
    )
    return parser.parse_args()


def on_mouse(event: int, x: int, y: int, flags: int, userdata: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"{POINT_LABELS[len(points) - 1]}: ({x}, {y})")


def draw_overlay(frame: np.ndarray) -> np.ndarray:
    display = frame.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 5, POINT_COLORS[i], -1)
        cv2.putText(
            display,
            f"{i + 1}:{POINT_LABELS[i]}",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            POINT_COLORS[i],
            2,
        )

    if len(points) < 4:
        prompt = f"Click {POINT_LABELS[len(points)]} ({len(points)}/4)"
    else:
        prompt = "4/4 selected. Press 'q' to save + quit, 'r' to reset."

    cv2.putText(display, prompt, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(
        display,
        "n=next frame  p=prev frame  r=reset  q=quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    return display


def load_frame(cap: cv2.VideoCapture, index: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(index, 0))
    ok, frame = cap.read()
    return frame if ok else None


def prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"Could not parse '{raw}' as a number, using default {default}.")
        return default


def save_result(lane_width_m: float, reference_distance_m: float) -> None:
    if len(points) != 4:
        print("\nLess than 4 points selected - nothing saved.")
        return

    homography_data = {
        "src": [[int(x), int(y)] for x, y in points],
        "lane_width_m": lane_width_m,
        "reference_distance_m": reference_distance_m,
    }
    update_calibration("homography", homography_data)

    near_left, near_right, far_left, far_right = points
    print(f"\nSaved to {CALIBRATION_PATH}:")
    print(f"  near-left={near_left}  near-right={near_right}")
    print(f"  far-left={far_left}  far-right={far_right}")
    print(f"  lane_width_m={lane_width_m}  reference_distance_m={reference_distance_m}")
    print(
        "\nobjectdetection.py will pick this up automatically on its next "
        "run - no copy-pasting needed."
    )


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    frame_index = args.start_frame
    frame = load_frame(cap, frame_index)
    if frame is None:
        raise RuntimeError(f"Could not read starting frame {frame_index} from {args.video_path}")

    window = "Calibration - click near-left, near-right, far-left, far-right"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_mouse)

    print("Click 4 points in order: near-left, near-right, far-left, far-right")
    print("These should be real, identifiable points on the road (e.g. where")
    print("lane markings cross a stop line, and the same lane edges further")
    print(f"down the road), where near-left/near-right are {args.lane_width}m apart")
    print(f"and far-left/far-right are the same edges {args.reference_distance}m further away.\n")

    while True:
        cv2.imshow(window, draw_overlay(frame))
        key = cv2.waitKey(20) & 0xFF

        if key in (ord("q"), 27):  # q or Esc
            break
        elif key == ord("r"):
            points.clear()
            print("\nPoints reset.")
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

    cap.release()
    cv2.destroyAllWindows()

    if len(points) == 4:
        print("\nEnter the real-world measurements for these 4 points")
        print("(press Enter to accept the default shown in brackets):\n")
        lane_width = prompt_float(
            "Lane width in meters (real distance between near-left/near-right, "
            "and between far-left/far-right)",
            args.lane_width,
        )
        reference_distance = prompt_float(
            "Reference distance in meters (real distance between the near "
            "line and the far line - e.g. counted dash+gap cycles x segment "
            "length)",
            args.reference_distance,
        )
    else:
        lane_width = args.lane_width
        reference_distance = args.reference_distance

    save_result(lane_width, reference_distance)


if __name__ == "__main__":
    main()
