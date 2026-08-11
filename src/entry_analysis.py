"""
Step 6: Detect the entry frame (when the wrist crosses the water line)
Step 7: Compute entry angle (body line vs. water line, direction-normalized)

Depends on:
  - pose_data.json   (from pose_extraction.py)
  - calibration.json (from calibration.py) -- must include water_line points
    or water_line_angle_rad, produced by your calibration tool

Usage:
    python src/entry_analysis.py outputs/pose_data.json data/calibration/calibration.json
"""

import argparse
import json
import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


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


def determine_dominant_side(pose_data: dict, landmark_pairs: dict, sample_frames: int = 30) -> dict:
    """
    landmark_pairs example: {"hip": ("left_hip", "right_hip"),
                              "shoulder": ("left_shoulder", "right_shoulder"),
                              "wrist": ("left_wrist", "right_wrist")}
    Returns dict mapping each body part -> the more visible side's landmark name,
    e.g. {"hip": "left_hip", "shoulder": "left_shoulder", "wrist": "left_wrist"}

    Checked once across sample_frames and applied consistently for the whole
    clip, so we don't jump between left/right landmarks frame to frame.
    """
    result = {}
    for part, (left_name, right_name) in landmark_pairs.items():
        vis_sum = {left_name: 0.0, right_name: 0.0}
        vis_count = {left_name: 0, right_name: 0}
        frames_checked = 0

        for frame in pose_data["frames"]:
            if frames_checked >= sample_frames:
                break
            lm_dict = frame["landmarks"]
            if lm_dict is None:
                continue
            for name in (left_name, right_name):
                if name in lm_dict:
                    vis_sum[name] += lm_dict[name]["visibility"]
                    vis_count[name] += 1
            frames_checked += 1

        avg_vis = {
            name: (vis_sum[name] / vis_count[name] if vis_count[name] > 0 else 0.0)
            for name in (left_name, right_name)
        }
        result[part] = max(avg_vis, key=avg_vis.get)

    return result


def _line_point_and_direction(calibration: dict):
    """
    Returns (point_on_line, unit_direction_vector) for the water line,
    reconstructed from calibration data. Expects calibration.json to have
    either:
      - "water_line_p1" and "water_line_p2" (raw clicked points), or
      - "water_line_angle_rad" plus some anchor point

    This assumes calibration.py was extended to save the raw click points.
    If your calibration.json only has water_line_angle_rad, this function
    falls back to using calibration_frame center as an anchor -- adjust if
    your calibration saves a specific anchor point instead.
    """
    if "water_line_p1" in calibration and "water_line_p2" in calibration:
        p1 = np.array(calibration["water_line_p1"], dtype=float)
        p2 = np.array(calibration["water_line_p2"], dtype=float)
        direction = p2 - p1
        direction = direction / np.linalg.norm(direction)
        return p1, direction

    # Fallback: reconstruct direction from angle only, anchored at
    # calibration["water_line_y_px"] if present (older calibration format),
    # using x=0 as an arbitrary anchor x-coordinate.
    angle = calibration["water_line_angle_rad"]
    anchor_y = calibration.get("water_line_y_px", 0.0)
    p1 = np.array([0.0, anchor_y])
    direction = np.array([np.cos(angle), np.sin(angle)])
    return p1, direction


def signed_distance_to_water_line(point: np.ndarray, line_point: np.ndarray,
                                   line_direction: np.ndarray) -> float:
    """
    Perpendicular signed distance from `point` to the water line.
    Positive = below the line (underwater side, since y increases downward),
    negative = above the line (in air).
    """
    # normal vector to the line (rotate direction by 90 degrees)
    normal = np.array([-line_direction[1], line_direction[0]])
    diff = point - line_point
    return float(np.dot(diff, normal))


def detect_entry_frame(pose_data: dict, calibration: dict, wrist_landmark: str) -> int:
    """
    Finds the first frame where the wrist's perpendicular distance to the
    (possibly tilted) water line crosses from negative (above/in-air) to
    positive (at/below the surface).
    """
    wrist_traj = get_xy_trajectory(pose_data, wrist_landmark)
    if len(wrist_traj) == 0:
        raise ValueError(f"No frames found with landmark '{wrist_landmark}' detected.")

    line_point, line_direction = _line_point_and_direction(calibration)

    prev_dist = None
    for row in wrist_traj:
        frame_idx, x, y = int(row[0]), row[1], row[2]
        dist = signed_distance_to_water_line(np.array([x, y]), line_point, line_direction)

        if prev_dist is not None and prev_dist < 0 <= dist:
            return frame_idx

        prev_dist = dist

    raise ValueError(
        "Could not detect a water-line crossing for this landmark. "
        "Check that the wrist is tracked through the entry, and that "
        "the water line calibration looks correct."
    )


def compute_entry_angle(pose_data: dict, calibration: dict, entry_frame: int,
                         wrist_landmark: str, shoulder_landmark: str, hip_landmark: str) -> dict:
    """
    At the entry frame, fits the body line through wrist -> shoulder -> hip
    and computes its angle relative to the water line (not image-horizontal).

    Direction-normalized: regardless of whether the dive travels left-to-right
    or right-to-left on screen, entry angle is reported the same way --
    as the acute angle between the body line and the water line, always
    measured as a positive value representing how steep the entry is.
    """
    frame = next((f for f in pose_data["frames"] if f["frame_idx"] == entry_frame), None)
    if frame is None or frame["landmarks"] is None:
        raise ValueError(f"No pose data available at entry frame {entry_frame}.")

    lm = frame["landmarks"]
    missing = [name for name in (wrist_landmark, shoulder_landmark, hip_landmark) if name not in lm]
    if missing:
        raise ValueError(f"Missing landmarks at entry frame {entry_frame}: {missing}")

    wrist = np.array([lm[wrist_landmark]["x_px"], lm[wrist_landmark]["y_px"]])
    hip = np.array([lm[hip_landmark]["x_px"], lm[hip_landmark]["y_px"]])

    # Body line vector: from hip to wrist (direction of travel into the water)
    body_vec = wrist - hip
    body_vec_unit = body_vec / np.linalg.norm(body_vec)

    _, water_direction = _line_point_and_direction(calibration)

    # Angle between body line and water line, via dot product.
    # Using abs() on the dot product and taking the acute angle makes this
    # direction-agnostic: left-to-right and right-to-left dives of the same
    # steepness report the same angle instead of supplementary angles.
    cos_angle = np.clip(abs(np.dot(body_vec_unit, water_direction)), -1.0, 1.0)
    angle_from_water_rad = np.arccos(cos_angle)
    # angle_from_water_rad: 0 = body line parallel to water (flat/shallow entry),
    # pi/2 = body line perpendicular to water (vertical/steep entry)

    angle_from_water_deg = float(np.degrees(angle_from_water_rad))

    return {
        "entry_frame": entry_frame,
        "entry_angle_deg": angle_from_water_deg,
        "note": "0 deg = parallel to water surface (flat entry), 90 deg = perpendicular (vertical entry)",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect entry frame and compute entry angle")
    parser.add_argument("pose_json", help="Path to pose_data.json")
    parser.add_argument("calibration_json", help="Path to calibration.json")
    args = parser.parse_args()

    pose_data = load_json(args.pose_json)
    calibration = load_json(args.calibration_json)

    sides = determine_dominant_side(pose_data, {
        "hip": ("left_hip", "right_hip"),
        "shoulder": ("left_shoulder", "right_shoulder"),
        "wrist": ("left_wrist", "right_wrist"),
    })
    print(f"Using landmarks: {sides}")

    entry_frame = detect_entry_frame(pose_data, calibration, sides["wrist"])
    print(f"Detected entry frame: {entry_frame}")

    result = compute_entry_angle(pose_data, calibration, entry_frame,
                                  sides["wrist"], sides["shoulder"], sides["hip"])
    print(json.dumps(result, indent=2))
