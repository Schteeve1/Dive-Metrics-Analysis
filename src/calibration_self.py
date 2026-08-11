"""
step 3 calibration

convert pixels to meters
"""

import cv2
import numpy as np
import json

def get_frame(video_path, frame_idx):

    frame = cv2.VideoCapture(video_path)
    if not frame.isOpened:
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get
    return
    
