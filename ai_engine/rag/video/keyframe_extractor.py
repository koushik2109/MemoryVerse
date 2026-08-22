"""
Video Keyframe Extractor
Uses OpenCV to extract one keyframe every 2 seconds from a video file.
Returns a list of (timestamp_sec, BGR numpy array) tuples.
"""
import cv2
import numpy as np
from typing import List, Tuple


def extract_keyframes(
    video_path: str,
    interval_sec: float = 2.0,
) -> Tuple[List[Tuple[float, np.ndarray]], float, float, int]:
    """
    Extract keyframes at a fixed time interval.

    Returns:
        keyframes       : list of (timestamp_sec, BGR frame)
        duration_sec    : total video duration
        fps             : frames per second
        total_keyframes : number of extracted keyframes
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec: float = total_frames / fps

    frame_interval = max(1, int(fps * interval_sec))
    keyframes: List[Tuple[float, np.ndarray]] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            timestamp = frame_idx / fps
            keyframes.append((timestamp, frame.copy()))
        frame_idx += 1

    cap.release()

    return keyframes, duration_sec, fps, len(keyframes)
