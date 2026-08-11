# Dive Analysis (v1)

A system that takes in a side-on video of a swimming dive and performs pose
extraction to detect body parts, then outputs metrics such as takeoff
velocity, reaction time, entry angle, and splash radius.

v1 scope: takeoff velocity, entry angle, splash radius, reaction time.

## Setup

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

**Step 1-2: pose extraction + trajectory plotting**
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

**Step 3: calibration**
```bash
python src/calibration.py data/raw/your_dive_clip.mp4 --known_length 5.0 --frame 0
```
Click 2 points spanning a known real-world length, then 2 points along the
water line, then press 's' to save.

**Step 4-5: takeoff frame + takeoff velocity**
```bash
python src/metrics.py outputs/pose_data.json data/calibration/calibration.json
```

**Step 6-7: entry frame + entry angle**
```bash
python src/entry_analysis.py outputs/pose_data.json data/calibration/calibration.json
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
│   ├── calibration.py       # Step 3: pixel-to-meter + water line calibration
│   ├── metrics.py           # Step 4-5: takeoff frame + takeoff velocity
│   ├── entry_analysis.py    # Step 6-7: entry frame + entry angle
│   └── splash_analysis.py   # Step 8: splash radius (frame-to-frame diffing)
├── outputs/             # cached pose data, plots, results
├── main.py              # pipeline entry point
└── requirements.txt
```


## Known limitations
- **Splash radius accuracy is depth-dependent.** Calibration uses a single
  meters-per-pixel scale factor from one reference measurement (e.g. lane
  rope width at one point in the frame). Because of camera perspective, the
  same real-world distance covers fewer pixels the farther it is from the
  camera -- so this scale factor is only strictly accurate at the depth it
  was measured at. If the splash occurs at a different depth in the pool
  than the calibration reference, reported splash radius will be biased
  (observed: visually cross-checking splash extent against the 5m lane
  rope suggested the reported value was an underestimate). A proper fix
  requires a homography-based perspective correction rather than a single
  scalar conversion factor.