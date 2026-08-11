"""
Entry point for the dive analysis pipeline.

v1 usage (Step 1-2 only for now — more stages will be wired in as you build them):

    python main.py --video data/raw/dive1.mp4

This will:
  1. Run MediaPipe Pose over every frame of the video
  2. Cache the extracted keypoints to outputs/pose_data.json
  3. Plot hip/ankle trajectories to outputs/trajectories.png for a sanity check
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pose_extraction import extract_pose_from_video, save_pose_data
from plot_trajectories import plot_trajectories


def main():
    parser = argparse.ArgumentParser(description="Swimming dive analysis pipeline (v1)")
    parser.add_argument("--video", required=True, help="Path to input dive video")
    parser.add_argument(
        "--pose_cache",
        default="outputs/pose_data.json",
        help="Where to cache extracted pose keypoints",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip MediaPipe and just re-plot from an existing pose_cache file",
    )
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)

    if not args.skip_extraction:
        print(f"Running MediaPipe Pose on {args.video} ...")
        pose_data = extract_pose_from_video(args.video)
        print(f"Processed {pose_data['frame_count']} frames at {pose_data['fps']:.1f} fps")
        save_pose_data(pose_data, args.pose_cache)
        print(f"Cached pose data to {args.pose_cache}")
    else:
        print(f"Skipping extraction, using cached pose data at {args.pose_cache}")

    plot_trajectories(args.pose_cache)


if __name__ == "__main__":
    main()
