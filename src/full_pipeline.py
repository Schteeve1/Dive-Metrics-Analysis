"""
Step 10: Full pipeline -- runs pose extraction, takeoff velocity, entry
angle, and splash radius in one pass, and writes a single JSON with all
results. Optionally produces an annotated output video marking the
detected takeoff frame, entry frame + entry angle line, and splash circle,
with a persistent metrics readout overlaid.

Usage:
    python src/full_pipeline.py data/raw/your_clip.mp4 data/calibration/calibration.json

    # also write an annotated video:
    python src/full_pipeline.py data/raw/your_clip.mp4 data/calibration/calibration.json --annotate
"""

import argparse
import json
import os

import cv2
import numpy as np

from pose_extraction import extract_pose_from_video, save_pose_data, load_pose_data
from metrics import (
    determine_dominant_side as determine_dominant_hip_side,
    get_xy_trajectory,
    detect_takeoff_frame,
    compute_takeoff_velocity,
)
from entry_analysis import (
    determine_dominant_side as determine_dominant_sides,
    detect_entry_frame,
    compute_entry_angle,
    load_json,
)
from splash_analysis import compute_splash_radius, get_entry_point_px


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(video_path: str, calibration_path: str,
                  pose_cache_path: str = "outputs/pose_data.json",
                  skip_extraction: bool = False,
                  velocity_threshold_px: float = 8.0) -> dict:

    if not skip_extraction:
        print(f"Running MediaPipe Pose on {video_path} ...")
        pose_data = extract_pose_from_video(video_path)
        os.makedirs(os.path.dirname(pose_cache_path) or ".", exist_ok=True)
        save_pose_data(pose_data, pose_cache_path)
        print(f"Cached pose data to {pose_cache_path}")
    else:
        pose_data = load_pose_data(pose_cache_path)

    calibration = load_json(calibration_path)

    # entry_analysis's version: picks best side for hip/shoulder/wrist together
    sides = determine_dominant_sides(pose_data, {
        "hip": ("left_hip", "right_hip"),
        "shoulder": ("left_shoulder", "right_shoulder"),
        "wrist": ("left_wrist", "right_wrist"),
    })
    print(f"Using landmarks: {sides}")

    # metrics.py's own dominant-hip logic (same averaging approach, computed
    # independently since compute_takeoff_velocity determines it internally
    # too) -- kept separate rather than forcing metrics.py's functions to
    # accept an externally-chosen landmark, since that would mean editing
    # your working, tested file just to satisfy this wrapper.
    dominant_hip = determine_dominant_hip_side(pose_data, ["left_hip", "right_hip"])
    hip_traj = get_xy_trajectory(pose_data, dominant_hip)
    takeoff_frame = detect_takeoff_frame(hip_traj, velocity_threshold_px=velocity_threshold_px)
    takeoff_result = compute_takeoff_velocity(pose_data, calibration, takeoff_frame)
    print(f"Takeoff frame: {takeoff_frame}, speed: {takeoff_result['takeoff_speed_mps']:.2f} m/s")

    entry_frame = detect_entry_frame(pose_data, calibration, sides["wrist"])
    entry_result = compute_entry_angle(pose_data, calibration, entry_frame,
                                        sides["wrist"], sides["shoulder"], sides["hip"])
    print(f"Entry frame: {entry_frame}, angle: {entry_result['entry_angle_deg']:.1f} deg")

    splash_result = compute_splash_radius(video_path, pose_data, calibration,
                                           entry_frame, sides["wrist"])
    print(f"Max splash radius: {splash_result['max_splash_radius_m']:.2f} m "
          f"at frame {splash_result['max_splash_frame']}")

    return {
        "video_path": video_path,
        "fps": pose_data["fps"],
        "landmarks_used": sides,
        "takeoff": takeoff_result,
        "entry": entry_result,
        "splash": splash_result,
    }


# ---------------------------------------------------------------------------
# Annotated video output
# ---------------------------------------------------------------------------

def write_annotated_video(video_path: str, results: dict, pose_data: dict,
                           out_path: str = "outputs/annotated.mp4") -> None:
    """
    Writes an annotated copy of the video:
      - persistent metrics readout in the top-left corner throughout
      - a marker + label at the takeoff frame
      - a marker + label at the entry frame
      - a circle at the max splash frame showing the detected splash radius
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    takeoff_frame = results["takeoff"]["takeoff_frame"]
    entry_frame = results["entry"]["entry_frame"]
    splash_frame = results["splash"]["max_splash_frame"]
    splash_radius_px = results["splash"]["max_splash_radius_px"]

    entry_point_px = get_entry_point_px(pose_data, entry_frame, results["landmarks_used"]["wrist"])

    summary_lines = [
        f"Takeoff speed: {results['takeoff']['takeoff_speed_mps']:.2f} m/s",
        f"Entry angle: {results['entry']['entry_angle_deg']:.1f} deg",
        f"Splash radius: {results['splash']['max_splash_radius_m']:.2f} m",
    ]

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # persistent metrics readout, top-left
        for i, line in enumerate(summary_lines):
            y = 30 + i * 28
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 0), 4, cv2.LINE_AA)  # black outline for readability
            cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, cv2.LINE_AA)

        if frame_idx == takeoff_frame:
            cv2.putText(frame, "TAKEOFF", (width // 2 - 60, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        if frame_idx == entry_frame:
            pt = (int(entry_point_px[0]), int(entry_point_px[1]))
            cv2.circle(frame, pt, 8, (0, 165, 255), -1)
            cv2.putText(frame, "ENTRY", (pt[0] + 12, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)

        if frame_idx == splash_frame:
            pt = (int(entry_point_px[0]), int(entry_point_px[1]))
            cv2.circle(frame, pt, int(splash_radius_px), (255, 0, 255), 2)
            cv2.putText(frame, "MAX SPLASH", (pt[0] + 12, pt[1] + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Saved annotated video to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full dive analysis pipeline")
    parser.add_argument("video_path", help="Path to the dive video")
    parser.add_argument("calibration_json", help="Path to calibration.json")
    parser.add_argument("--pose_cache", default="outputs/pose_data.json",
                        help="Where to cache/read extracted pose keypoints")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip MediaPipe and reuse cached pose data")
    parser.add_argument("--velocity_threshold_px", type=float, default=8.0)
    parser.add_argument("--annotate", action="store_true",
                        help="Also write an annotated output video")
    parser.add_argument("--results_out", default="outputs/results.json",
                        help="Where to save the combined metrics JSON")
    args = parser.parse_args()

    results = run_pipeline(
        args.video_path, args.calibration_json,
        pose_cache_path=args.pose_cache,
        skip_extraction=args.skip_extraction,
        velocity_threshold_px=args.velocity_threshold_px,
    )

    os.makedirs(os.path.dirname(args.results_out) or ".", exist_ok=True)
    with open(args.results_out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved combined results to {args.results_out}")

    if args.annotate:
        pose_data = load_pose_data(args.pose_cache)
        write_annotated_video(args.video_path, results, pose_data)
