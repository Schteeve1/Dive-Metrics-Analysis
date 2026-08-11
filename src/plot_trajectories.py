"""
Step 2 checkpoint: plot hip and ankle y-position over time from cached
pose data. You're looking for a visible three-phase shape:
  flat (on block) -> rise (flight) -> drop (entry into water)

If you don't see that shape clearly, pose tracking isn't clean enough yet
and you should fix that before building event detection on top of it.
"""

import sys
import matplotlib.pyplot as plt

from pose_extraction import load_pose_data, get_trajectory


def plot_trajectories(pose_json_path: str, out_image_path: str = "outputs/trajectories.png"):
    pose_data = load_pose_data(pose_json_path)

    # Average left/right hip and ankle to get a rough centerline trajectory
    hip_traj = get_trajectory(pose_data, "left_hip")
    ankle_traj = get_trajectory(pose_data, "left_ankle")

    plt.figure(figsize=(10, 6))

    if len(hip_traj) > 0:
        plt.plot(hip_traj[:, 0], hip_traj[:, 1], label="Left hip y (px)")
    if len(ankle_traj) > 0:
        plt.plot(ankle_traj[:, 0], ankle_traj[:, 1], label="Left ankle y (px)")

    # Note: image y-axis increases downward, so "rising" in the dive
    # actually means y DECREASING on this plot. Invert axis for intuitive reading.
    plt.gca().invert_yaxis()
    plt.xlabel("Frame index")
    plt.ylabel("Pixel y-position (inverted: up = up)")
    plt.title("Hip / Ankle trajectory over the dive")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_image_path, dpi=150)
    print(f"Saved trajectory plot to {out_image_path}")
    plt.show()
    sys.exit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_trajectories.py <pose_data.json>")
        sys.exit(1)
    plot_trajectories(sys.argv[1])
