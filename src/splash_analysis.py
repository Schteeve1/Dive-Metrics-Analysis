"""
Step 8: Splash radius.

Approach:
  1. Grab a "calm water" reference frame from just before entry
  2. For each frame in a window after entry, diff against that reference
  3. Threshold the diff to isolate what changed (the splash)
  4. Find contours in the thresholded mask, keep the one closest to the
     entry point (filters out unrelated motion elsewhere in frame)
  5. Track the largest splash radius across the window, convert px -> m

Needs the actual video file (not just pose_data.json), since splash
detection works on raw pixel differences, not pose keypoints.

Usage:
    python src/splash_analysis.py data/raw/your_clip.mp4 outputs/pose_data.json \
        data/calibration/calibration.json --entry_frame 73 --wrist_landmark left_wrist
"""

import argparse
import json

import cv2
import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_frame_at_index(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise IOError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def get_entry_point_px(pose_data: dict, entry_frame: int, wrist_landmark: str) -> tuple:
    """Pixel (x, y) of the wrist at the entry frame -- used to filter
    which contour in each diff frame is actually the splash, versus
    unrelated motion elsewhere (other swimmers, background movement)."""
    frame = next((f for f in pose_data["frames"] if f["frame_idx"] == entry_frame), None)
    if frame is None or frame["landmarks"] is None or wrist_landmark not in frame["landmarks"]:
        raise ValueError(f"No '{wrist_landmark}' landmark data at entry frame {entry_frame}.")
    lm = frame["landmarks"][wrist_landmark]
    return (lm["x_px"], lm["y_px"])


def compute_splash_radius_from_frames(frames: list, frame_indices: list,
                                       entry_point_px: tuple, meters_per_pixel: float,
                                       diff_threshold: int = 30, min_contour_area_px: float = 20.0,
                                       max_distance_from_entry_px: float = 400.0,
                                       blur_ksize: int = 5) -> dict:
    """
    Core splash detection logic, separated from video I/O so it can be
    tested with synthetic frames directly.

    Uses FRAME-TO-FRAME differencing (each frame vs. the one immediately
    before it), not a single fixed "calm water" reference. A fixed
    reference accumulates every change since that moment -- the swimmer
    continuing to swim forward, ripples spreading pool-wide -- so the
    measured region only ever grows and never shows the splash receding.
    Frame-to-frame diffing measures "how much is changing right now",
    so once the water and swimmer settle, the diff naturally shrinks,
    giving a genuine rise-then-fall curve.

    frames: list of BGR frames (numpy arrays), including ONE extra frame
        before the analysis window starts (used as the first diff's
        predecessor) -- so len(frames) == len(frame_indices) + 1
    frame_indices: frame indices for the window being analyzed (NOT
        including that leading extra frame)
    entry_point_px: (x, y) approximate splash origin, used to filter contours
    """
    if len(frames) != len(frame_indices) + 1:
        raise ValueError(
            f"Expected {len(frame_indices) + 1} frames (window + 1 leading "
            f"frame for the first diff), got {len(frames)}."
        )

    def prep(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    prev_gray = prep(frames[0])

    max_radius_px = 0.0
    max_radius_frame = None
    per_frame_results = []

    for frame, frame_idx in zip(frames[1:], frame_indices):
        gray = prep(frame)

        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_radius_px = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area_px:
                continue

            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            dist_from_entry = np.hypot(cx - entry_point_px[0], cy - entry_point_px[1])

            if dist_from_entry > max_distance_from_entry_px:
                continue  # not near the entry point, likely unrelated motion

            if radius > best_radius_px:
                best_radius_px = radius

        per_frame_results.append({"frame_idx": frame_idx, "splash_radius_px": best_radius_px})

        if best_radius_px > max_radius_px:
            max_radius_px = best_radius_px
            max_radius_frame = frame_idx

        prev_gray = gray  # roll the reference forward each frame

    return {
        "max_splash_radius_px": max_radius_px,
        "max_splash_radius_m": max_radius_px * meters_per_pixel,
        "max_splash_frame": max_radius_frame,
        "per_frame": per_frame_results,
    }


def compute_splash_radius(video_path: str, pose_data: dict, calibration: dict,
                           entry_frame: int, wrist_landmark: str,
                           window_frames: int = 20, diff_threshold: int = 30) -> dict:
    """
    Full pipeline: reads the actual video frames needed, then delegates
    to compute_splash_radius_from_frames for the core detection logic.
    Fetches one extra leading frame (entry_frame - 1) so the very first
    frame in the window has something to diff against.
    """
    start_idx = entry_frame - 1
    if start_idx < 0:
        raise ValueError(f"entry_frame ({entry_frame}) is too early to fetch a leading frame.")

    frame_indices = list(range(entry_frame, entry_frame + window_frames))
    all_indices_needed = [start_idx] + frame_indices
    frames = [get_frame_at_index(video_path, idx) for idx in all_indices_needed]

    entry_point_px = get_entry_point_px(pose_data, entry_frame, wrist_landmark)
    meters_per_pixel = calibration["meters_per_pixel"]

    return compute_splash_radius_from_frames(
        frames, frame_indices, entry_point_px,
        meters_per_pixel, diff_threshold=diff_threshold,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute splash radius after water entry")
    parser.add_argument("video_path", help="Path to the dive video")
    parser.add_argument("pose_json", help="Path to pose_data.json")
    parser.add_argument("calibration_json", help="Path to calibration.json")
    parser.add_argument("--entry_frame", type=int, required=True,
                        help="Entry frame index, from entry_analysis.py output")
    parser.add_argument("--wrist_landmark", default="left_wrist",
                        help="Which wrist landmark to use as the splash origin reference "
                             "(use the same one entry_analysis.py reported as dominant)")
    parser.add_argument("--window_frames", type=int, default=20,
                        help="How many frames after entry to scan for max splash size")
    parser.add_argument("--diff_threshold", type=int, default=30,
                        help="Pixel intensity difference to count as 'changed' (tune this per video)")
    args = parser.parse_args()

    pose_data = load_json(args.pose_json)
    calibration = load_json(args.calibration_json)

    result = compute_splash_radius(
        args.video_path, pose_data, calibration,
        args.entry_frame, args.wrist_landmark,
        window_frames=args.window_frames,
        diff_threshold=args.diff_threshold,
    )

    print(f"Max splash radius: {result['max_splash_radius_m']:.3f} m "
          f"({result['max_splash_radius_px']:.1f} px) at frame {result['max_splash_frame']}")
    print(json.dumps(result, indent=2))
