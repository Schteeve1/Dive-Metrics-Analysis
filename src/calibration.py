"""
Step 3: Camera calibration.

Run this once per video/camera setup (if your camera position changes,
re-run it). It opens a single frame from your video and lets you click:

  1. Two points spanning a KNOWN real-world length (e.g. the pool lane
     rope spacing, or block width) -> used to compute meters_per_pixel
  2. One point on the water surface line -> used later for entry detection

Controls:
  - Click point 1 and point 2 of your reference length (in order)
  - Click one point on the water line
  - Press 's' to save and quit
  - Press 'r' to reset your clicks and try again
  - Press 'q' to quit without saving

Usage:
    python src/calibration.py data/raw/your_clip.mp4 --known_length 2.5 --frame 0
"""

import argparse
import json
import os
import numpy as np

import cv2

clicked_points = []
WINDOW_NAME = "Calibration - click ref point 1, ref point 2, then water line"
DISPLAY_SCALE = 0.5


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        original_x = x / DISPLAY_SCALE
        original_y = y / DISPLAY_SCALE
        clicked_points.append((original_x, original_y))
        print(f"Point {len(clicked_points)} clicked: ({x}, {y})")


def get_frame(video_path: str, frame_idx: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def run_calibration(video_path: str, known_length_m: float, frame_idx: int = 0,
                     out_path: str = "data/calibration/calibration.json"):
    frame = get_frame(video_path, frame_idx)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    print("\nInstructions:")
    print(f"  1. Click two points spanning your known {known_length_m}m reference")
    print("  2. Click two points on the water surface line")
    print("  3. Press 's' to save, 'r' to reset clicks, 'q' to quit without saving\n")


    while True:
        display_frame = cv2.resize(frame, None, fx = DISPLAY_SCALE, fy = DISPLAY_SCALE)
        for i, pt in enumerate(clicked_points):
            draw_pt = (int(pt[0] * DISPLAY_SCALE), int(pt[1] * DISPLAY_SCALE) )
            cv2.circle(display_frame, draw_pt, 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i + 1), (draw_pt[0] + 8, draw_pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
        cv2.imshow(WINDOW_NAME, display_frame)


        key = cv2.waitKey(20) & 0xFF

        if key == ord('r'):
            clicked_points.clear()
            print("Reset. Click again.")

        elif key == ord('q'):
            print("Quit without saving.")
            cv2.destroyAllWindows()
            return

        elif key == ord('s'):
            if len(clicked_points) < 3:
                print(f"Need 3 points (2 reference + 1 water line), you have {len(clicked_points)}. Keep clicking.")
                continue
                
            (x1, y1), (x2, y2) = clicked_points[0], clicked_points[1]
            pixel_distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            meters_per_pixel = known_length_m / pixel_distance

            water_line_dx = clicked_points[3][0] - clicked_points[2][0]
            water_line_dy = clicked_points[3][1] - clicked_points[2][1]
            if water_line_dx < 0:
                water_line_dx = -water_line_dx
                water_line_dy = -water_line_dy
            water_line_angle_rad = np.arctan2(water_line_dy, water_line_dx)

            calibration = {
                "video_path": video_path,
                "calibration_frame_idx": frame_idx,
                "known_length_m": known_length_m,
                "reference_pixel_distance": pixel_distance,
                "meters_per_pixel": meters_per_pixel,
                "water_line_angle_rad": water_line_angle_rad,
            }

            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(calibration, f, indent=2)

            print(f"\nSaved calibration to {out_path}:")
            print(json.dumps(calibration, indent=2))
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive camera calibration")
    parser.add_argument("video_path", help="Path to the dive video")
    parser.add_argument("--known_length", type=float, required=True,
                        help="Real-world length in meters of your reference (e.g. lane rope spacing)")
    parser.add_argument("--frame", type=int, default=0,
                        help="Frame index to calibrate on (pick one where your reference is clearly visible)")
    parser.add_argument("--out", default="data/calibration/calibration.json",
                        help="Where to save the calibration JSON")
    args = parser.parse_args()

    run_calibration(args.video_path, args.known_length, args.frame, args.out)
