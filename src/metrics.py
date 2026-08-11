"""
Step 4: Detect the takeoff frame (last frame before the swimmer leaves the block)
Step 5: Compute takeoff velocity using hip trajectory + calibration data

Usage:
    python src/metrics.py outputs/pose_data.json data/calibration/calibration.json
"""

import argparse
import json
import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def determine_dominant_side(pose_data: dict, landmark_names: list, sample_frames: int = 30) -> str:
    """
    Checks average visibility across the first `sample_frames` valid frames
    for each candidate landmark, returns the name of whichever is most
    reliably visible for this clip.
    """
    visibility_sums = {name: 0.0 for name in landmark_names}
    visibility_counts = {name: 0 for name in landmark_names}

    frames_checked = 0
    for frame in pose_data["frames"]:
        if frames_checked >= sample_frames:
            break
        lm_dict = frame["landmarks"]
        if lm_dict is None:
            continue

        for name in landmark_names:
            if name in lm_dict:
                visibility_sums[name] += lm_dict[name]["visibility"]
                visibility_counts[name] += 1

        frames_checked += 1

    avg_visibility = {
        name: (visibility_sums[name] / visibility_counts[name] if visibility_counts[name] > 0 else 0.0)
        for name in landmark_names
    }

    best_name = max(avg_visibility, key=avg_visibility.get)
    return best_name


def get_xy_trajectory(pose_data: dict, landmark_name: str) -> np.ndarray:
    """Returns an (N, 3) array of [frame_idx, x_px, y_px], skipping frames
    where the landmark wasn't detected."""
    rows = []
    for frame in pose_data["frames"]:
        lm_dict = frame["landmarks"]
        if lm_dict is not None and landmark_name in lm_dict:
            lm = lm_dict[landmark_name]

            rows.append([frame["frame_idx"], lm["x_px"], lm["y_px"]])
    return np.array(rows)


def detect_takeoff_frame(hip_traj: np.ndarray, velocity_threshold_px: float = 8.0,
                          sustained_frames: int = 4) -> int:
    """
    Finds the first frame where frame-to-frame hip displacement exceeds
    velocity_threshold_px and stays elevated for `sustained_frames` in a row.
    Returns the frame index just BEFORE that sustained movement starts
    (i.e. the last "still on the block" frame).

    velocity_threshold_px is in pixels/frame -- you will likely need to tune
    this per video based on resolution and how far the swimmer is from camera.
    """
    frame_idxs = hip_traj[:, 0].astype(int)
    xy = hip_traj[:, 1:3]

    # frame-to-frame displacement magnitude
    displacements = np.linalg.norm(np.diff(xy, axis=0), axis=1)

    for i in range(len(displacements) - sustained_frames):
        window = displacements[i:i + sustained_frames]
        if np.all(window > velocity_threshold_px):
            # movement starts at displacement index i, which is between
            # frame_idxs[i] and frame_idxs[i+1] -- so the last "still" frame
            # is frame_idxs[i]
            return int(frame_idxs[i])

    raise ValueError(
        "Could not detect a sustained-movement takeoff point. "
        "Try lowering velocity_threshold_px or check the trajectory plot "
        "to see if pose tracking is clean during the takeoff phase."
    )


def compute_takeoff_velocity(pose_data: dict, calibration: dict,
                              takeoff_frame: int, lookforward_frames: int = 6) -> dict:
    """
    Fits a linear regression to hip x and y position over the
    `lookforward_frames` frames AFTER (and including) takeoff_frame,
    converts the slope (px/frame) to m/s using calibration + fps.

    We use frames AFTER takeoff rather than before: once airborne, the
    swimmer is in projectile motion (only gravity acting on them), which
    gives a clean, physically well-defined trajectory to fit. The frames
    right before takeoff are contaminated by block-contact motion and can
    be near-stationary right up until the last instant, making them a poor
    basis for a velocity estimate.
    """
    dominant_hip = determine_dominant_side(pose_data, ["left_hip", "right_hip"])
    print(f"Using {dominant_hip} for takeoff velocity computation (more consistently visible)")
    hip_traj = get_xy_trajectory(pose_data, dominant_hip)

    # Select frames in [takeoff_frame, takeoff_frame + lookforward_frames]
    mask = (hip_traj[:, 0] >= takeoff_frame) & (hip_traj[:, 0] <= takeoff_frame + lookforward_frames)
    window = hip_traj[mask]

    if len(window) < 3:
        raise ValueError(
            f"Only {len(window)} valid frames found in the lookforward window -- "
            "need at least 3 for a reliable fit. Pose tracking may have gaps "
            "here (e.g. splash occlusion cutting the window short); check "
            "pose_data.json around this frame range."
        )

    frames = window[:, 0]
    xs = window[:, 1]
    ys = window[:, 2]

    # Linear fit: position = slope * frame + intercept
    vx_slope, _ = np.polyfit(frames, xs, 1)  # px per frame
    vy_slope, _ = np.polyfit(frames, ys, 1)  # px per frame

    fps = pose_data["fps"]
    meters_per_pixel = calibration["meters_per_pixel"]

    vx_mps = vx_slope * fps * meters_per_pixel
    vy_mps = vy_slope * fps * meters_per_pixel

    # Note: image y increases downward, so a "forward and slightly downward"
    # takeoff will show vy as positive here. Flip sign if you want "up" positive.
    speed_mps = float(np.sqrt(vx_mps**2 + vy_mps**2))

    return {
        "takeoff_frame": takeoff_frame,
        "frames_used_in_fit": int(len(window)),
        "vx_mps": float(vx_mps),
        "vy_mps": float(vy_mps),
        "takeoff_speed_mps": speed_mps,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect takeoff frame and compute takeoff velocity")
    parser.add_argument("pose_json", help="Path to pose_data.json")
    parser.add_argument("calibration_json", help="Path to calibration.json")
    parser.add_argument("--velocity_threshold_px", type=float, default=8.0,
                        help="Frame-to-frame pixel displacement to count as 'moving' (tune this)")
    parser.add_argument("--lookforward_frames", type=int, default=6,
                        help="How many frames after takeoff (airborne) to use for the velocity fit")
    args = parser.parse_args()

    pose_data = load_json(args.pose_json)
    calibration = load_json(args.calibration_json)

    dominant_hip = determine_dominant_side(pose_data, ["left_hip", "right_hip"])
    print(f"Using {dominant_hip} for this clip (more consistently visible)")

    hip_traj = get_xy_trajectory(pose_data, dominant_hip)
    takeoff_frame = detect_takeoff_frame(hip_traj, velocity_threshold_px=args.velocity_threshold_px)
    print(f"Detected takeoff frame: {takeoff_frame}")

    result = compute_takeoff_velocity(pose_data, calibration, takeoff_frame,
                                       lookforward_frames=args.lookforward_frames)
    print(json.dumps(result, indent=2))
