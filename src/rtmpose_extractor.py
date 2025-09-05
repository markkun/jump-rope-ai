# src/rtmpose_extractor.py
import cv2
import numpy as np
import os
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules
import yaml

register_all_modules()

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

class RTMPoseExtractor:
    def __init__(self):
        self.model = init_model(
            cfg['rtmpose']['config'],
            cfg['rtmpose']['checkpoint'],
            device='cuda:0'
        )

    def extract(self, video_path, output_npy):
        cap = cv2.VideoCapture(video_path)
        pose_sequence = []

        while True:
            ret, frame = cap.read()
            if not ret: break

            results = inference_topdown(self.model, frame)
            if len(results) > 0:
                kpts = results[0].pred_instances.keypoints[0]  # (17,2)
                scores = results[0].pred_instances.keypoint_scores[0]  # (17,)
                person_kpts = np.hstack([kpts, scores.reshape(-1,1)])  # (17,3)
                pose_sequence.append(person_kpts)
            else:
                pose_sequence.append(np.zeros((17,3)))

        cap.release()
        pose_data = np.array(pose_sequence)
        np.save(output_npy, pose_data)
        print(f"Saved: {output_npy}, shape: {pose_data.shape}")
