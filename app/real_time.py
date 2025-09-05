# app/real_time.py
import cv2
from src.rtmpose_extractor import RTMPoseExtractor
from src.counter import count_jumps
import numpy as np

extractor = RTMPoseExtractor()
cap = cv2.VideoCapture(0)
buffer = []

while True:
    ret, frame = cap.read()
    if not ret: break

    results = extractor.model(frame)
    if len(results) > 0:
        kpts = results[0].pred_instances.keypoints[0]
        scores = results[0].pred_instances.keypoint_scores[0]
        buffer.append(np.hstack([kpts, scores.reshape(-1,1)]))

    if len(buffer) >= 300:
        buffer = buffer[-300:]
        data = np.array(buffer)
        count, peaks, _ = count_jumps(data)
        print(f"实时计数: {count}")

    cv2.imshow('Jump Rope AI', frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
cv2.destroyAllWindows()
