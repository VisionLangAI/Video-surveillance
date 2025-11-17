# preprocessing/keyframe_extractor.py

import cv2
import os

def extract_keyframes(video_path, output_dir, method="uniform", k=10):
    """
    Extract key frames from video.
    :param video_path: path to input video
    :param output_dir: where to save extracted frames
    :param method: 'uniform' or 'diff'
    :param k: number of key frames to extract (if uniform)
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if method == "uniform":
        # pick k frames evenly spaced
        indices = [int(frame_count * i / k) for i in range(k)]
    elif method == "diff":
        # difference-based: pick frames with largest difference
        # read all frames -> compute histogram diff
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        # compute frame differences
        diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i-1])
            diffs.append(diff.sum())
        # pick top k difference frames
        indices = sorted(range(1, len(diffs)), key=lambda i: diffs[i-1], reverse=True)[:k]
    else:
        raise ValueError("Unknown method for keyframe extraction")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    cap.release()
