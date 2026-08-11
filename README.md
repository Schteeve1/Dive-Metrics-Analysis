# Dive Analysis (v1)

Automated swimming dive biomechanics analysis from a single fixed-angle video.
v1 scope: takeoff velocity, entry angle, splash radius, reaction time.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage (current: Step 1-2, pose extraction + trajectory plotting)

```bash
python main.py --video data/raw/your_dive_clip.mp4
```

This will:
1. Run MediaPipe Pose over every frame
2. Cache keypoints to `outputs/pose_data.json` (so you don't need to
   re-run MediaPipe every time you tweak downstream logic)
3. Plot hip/ankle trajectories to `outputs/trajectories.png`

To re-plot from cached data without re-running MediaPipe:

```bash
python main.py --video data/raw/your_dive_clip.mp4 --skip_extraction
```

## Project structure

```
dive-analysis/
├── data/
│   ├── raw/            # input video clips go here
│   └── calibration/    # reference measurements (pixel-to-meter, water line)
├── src/
│   ├── pose_extraction.py   # Step 1-2: video load + MediaPipe pose
│   ├── plot_trajectories.py # Step 2 checkpoint: visualize trajectories
│   ├── calibration.py        # Step 3 (TODO)
│   ├── event_detection.py    # Step 4/6/9 (TODO)
│   └── metrics.py            # Step 5/7/8 (TODO)
├── outputs/             # cached pose data, plots, results
├── main.py              # pipeline entry point
└── requirements.txt
```

## Build roadmap

- [x] Step 1-2: Load video, extract pose, plot raw trajectories
- [ ] Step 3: Camera calibration (pixel -> meter, water line y-coordinate)
- [ ] Step 4: Detect takeoff frame
- [ ] Step 5: Compute takeoff velocity
- [ ] Step 6: Detect entry frame
- [ ] Step 7: Compute entry angle
- [ ] Step 8: Splash radius (frame diff + contours)
- [ ] Step 9: Reaction time (audio start signal + first movement)
- [ ] Step 10: Wire into one pipeline output (JSON + optional annotated video)

## Known limitations (v1)

- Single fixed camera, assumes minimal perspective distortion
  (camera roughly perpendicular to the dive plane)
- Calibration is a manual one-time pixel-to-meter measurement, not full 3D
- Reaction time precision is bounded by video frame rate — 30fps gives
  ~33ms resolution, which is coarse relative to typical human variation
