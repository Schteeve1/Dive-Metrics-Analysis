"""
Step 9: Reaction time.

Two sub-detections:
  1. Start signal timestamp -- the starting beep, found via audio amplitude
     onset detection (looks for the rising edge of the loudest sound event,
     assumed to be the starting signal).
  2. First movement frame -- using the same "sustained displacement above
     threshold" pattern as takeoff detection, but searching forward from
     the start signal's timestamp rather than from frame 0.

reaction_time = first_movement_timestamp - start_signal_timestamp

Requires ffmpeg to be installed and on PATH (librosa uses it under the hood
to read audio out of video files). If librosa.load fails with a decoding
error, that's almost always a missing ffmpeg install, not a bug here.

Usage:
    python src/reaction_analysis.py data/raw/your_clip.mp4 outputs/pose_data.json
"""

import argparse
import json

import numpy as np
import librosa


def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_xy_trajectory(pose_data: dict, landmark_name: str) -> np.ndarray:
    rows = []
    for frame in pose_data["frames"]:
        lm_dict = frame["landmarks"]
        if lm_dict is not None and landmark_name in lm_dict:
            lm = lm_dict[landmark_name]
            rows.append([frame["frame_idx"], lm["x_px"], lm["y_px"]])
    return np.array(rows)


def determine_dominant_side(pose_data: dict, left_name: str, right_name: str,
                             sample_frames: int = 30) -> str:
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
    return max(avg_vis, key=avg_vis.get)


def _detect_onset_from_waveform(amplitude: np.ndarray, sr: int,
                                 onset_fraction: float = 0.3,
                                 min_sustain_ms: float = 5.0) -> float:
    """
    Core onset-detection logic, separated from audio loading so it can be
    tested with synthetic waveforms directly.

    Finds the rising edge of the loudest sound event in the clip: takes
    `onset_fraction` of the peak amplitude as a threshold, then finds the
    first point where amplitude crosses that threshold and stays elevated
    for `min_sustain_ms` (avoids triggering on a single noisy sample).

    Assumes the starting beep/buzzer is the loudest sound in the clip --
    true for most dive-start footage, but worth checking if your audio has
    louder background noise (shouting, echo, music) than the beep itself.
    """
    peak_amp = amplitude.max()
    if peak_amp <= 0:
        raise ValueError("Audio track appears to be silent (max amplitude is 0).")

    threshold = peak_amp * onset_fraction
    min_sustain_samples = max(1, int(sr * min_sustain_ms / 1000))

    for i in range(len(amplitude) - min_sustain_samples):
        if np.all(amplitude[i:i + min_sustain_samples] > threshold):
            return i / sr

    raise ValueError(
        "Could not detect a clear audio onset. Try lowering onset_fraction, "
        "or check that the clip actually has an audible starting signal."
    )


def detect_start_signal(video_path: str, onset_fraction: float = 0.3,
                         min_sustain_ms: float = 5.0) -> float:
    """Loads audio from the video and returns the start signal's timestamp (s)."""
    y, sr = librosa.load(video_path, sr=None, mono=True)
    amplitude = np.abs(y)
    return _detect_onset_from_waveform(amplitude, sr, onset_fraction, min_sustain_ms)


def detect_first_movement_frame(traj: np.ndarray, search_start_frame: int,
                                 fps: float, velocity_threshold_px: float = 6.0,
                                 sustained_frames: int = 3) -> int:
    """
    Same "sustained displacement above threshold" pattern as takeoff
    detection, but only searching frames at/after search_start_frame
    (the start-signal timestamp converted to a frame index), so early
    pre-signal fidgeting on the blocks doesn't false-trigger this.
    """
    mask = traj[:, 0] >= search_start_frame
    sub_traj = traj[mask]

    if len(sub_traj) < sustained_frames + 1:
        raise ValueError(
            f"Not enough tracked frames after frame {search_start_frame} "
            "to detect movement. Check pose tracking quality in this range."
        )

    frame_idxs = sub_traj[:, 0].astype(int)
    xy = sub_traj[:, 1:3]
    displacements = np.linalg.norm(np.diff(xy, axis=0), axis=1)

    for i in range(len(displacements) - sustained_frames):
        window = displacements[i:i + sustained_frames]
        if np.all(window > velocity_threshold_px):
            return int(frame_idxs[i])

    raise ValueError(
        "Could not detect sustained first movement after the start signal. "
        "Try lowering velocity_threshold_px."
    )


def compute_reaction_time(video_path: str, pose_data: dict,
                           onset_fraction: float = 0.3,
                           velocity_threshold_px: float = 6.0) -> dict:
    fps = pose_data["fps"]

    start_signal_sec = detect_start_signal(video_path, onset_fraction=onset_fraction)
    start_signal_frame = int(round(start_signal_sec * fps))

    dominant_ankle = determine_dominant_side(pose_data, "left_ankle", "right_ankle")
    ankle_traj = get_xy_trajectory(pose_data, dominant_ankle)

    first_movement_frame = detect_first_movement_frame(
        ankle_traj, start_signal_frame, fps, velocity_threshold_px=velocity_threshold_px
    )
    first_movement_sec = first_movement_frame / fps

    reaction_time_sec = first_movement_sec - start_signal_sec

    return {
        "start_signal_sec": start_signal_sec,
        "start_signal_frame": start_signal_frame,
        "first_movement_frame": first_movement_frame,
        "first_movement_sec": first_movement_sec,
        "reaction_time_sec": reaction_time_sec,
        "landmark_used": dominant_ankle,
        "note": "Precision is bounded by video fps -- frame resolution = 1/fps seconds",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute reaction time from start signal to first movement")
    parser.add_argument("video_path", help="Path to the dive video (must have audio)")
    parser.add_argument("pose_json", help="Path to pose_data.json")
    parser.add_argument("--onset_fraction", type=float, default=0.3,
                        help="Fraction of peak audio amplitude to count as the start-signal threshold")
    parser.add_argument("--velocity_threshold_px", type=float, default=6.0,
                        help="Frame-to-frame pixel displacement to count as 'moving' (tune this)")
    args = parser.parse_args()

    pose_data = load_json(args.pose_json)

    result = compute_reaction_time(
        args.video_path, pose_data,
        onset_fraction=args.onset_fraction,
        velocity_threshold_px=args.velocity_threshold_px,
    )

    print(f"Reaction time: {result['reaction_time_sec']*1000:.1f} ms")
    print(json.dumps(result, indent=2))
