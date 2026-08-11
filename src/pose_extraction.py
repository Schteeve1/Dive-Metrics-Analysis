"""
Step 1-2: Load a video, run MediaPipe Pose on every frame, and return/save
per-frame keypoints so later pipeline stages can work on cached data instead
of re-running pose estimation every time.
"""

import json
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# Landmark indices we care about (MediaPipe's fixed 33-point body model).
# Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
LANDMARKS_OF_INTEREST = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def extract_pose_from_video(video_path: str, model_complexity: int = 1) -> dict:
    """
    Runs MediaPipe Pose over every frame of a video.

    Returns a dict:
    {
        "fps": float,
        "width": int,
        "height": int,
        "frame_count": int,
        "frames": [
            {
                "frame_idx": int,
                "landmarks": {
                    "left_hip": {"x": float, "y": float, "z": float, "visibility": float},
                    ...
                } or None if no pose detected in this frame
            },
            ...
        ]
    }

    x, y are normalized [0, 1] by MediaPipe; we also store pixel versions
    (x_px, y_px) using the video's actual width/height, since that's what
    later calibration/metric code will want to use.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frames_data = []

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe expects RGB, OpenCV gives BGR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = {}
                all_lm = results.pose_landmarks.landmark
                for name, idx in LANDMARKS_OF_INTEREST.items():
                    lm = all_lm[idx]
                    landmarks[name] = {
                        "x": lm.x,
                        "y": lm.y,
                        "x_px": lm.x * width,
                        "y_px": lm.y * height,
                        "z": lm.z,
                        "visibility": lm.visibility,
                    }
            else:
                landmarks = None

            frames_data.append({"frame_idx": frame_idx, "landmarks": landmarks})
            frame_idx += 1

    cap.release()

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "frames": frames_data,
    }


def save_pose_data(pose_data: dict, out_path: str) -> None:
    """Cache extracted pose data to JSON so you don't need to re-run
    MediaPipe every time you tweak downstream logic."""
    with open(out_path, "w") as f:
        json.dump(pose_data, f)


def load_pose_data(in_path: str) -> dict:
    with open(in_path, "r") as f:
        return json.load(f)


def get_trajectory(pose_data: dict, landmark_name: str) -> np.ndarray:
    """
    Returns an (N, 2) array of [frame_idx, y_px] for a given landmark,
    skipping frames where the landmark wasn't detected.
    Useful for plotting / event detection on a single coordinate.
    """
    rows = []
    for frame in pose_data["frames"]:
        if frame["landmarks"] is not None and landmark_name in frame["landmarks"]:
            lm = frame["landmarks"][landmark_name]
            rows.append([frame["frame_idx"], lm["y_px"]])
    return np.array(rows)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pose_extraction.py <video_path> [output_json_path]")
        sys.exit(1)

    video_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/pose_data.json"

    print(f"Running MediaPipe Pose on {video_path} ...")
    pose_data = extract_pose_from_video(video_path)
    print(f"Processed {pose_data['frame_count']} frames at {pose_data['fps']:.1f} fps")

    save_pose_data(pose_data, out_path)
    print(f"Saved pose data to {out_path}")
