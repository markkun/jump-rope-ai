# src/multi_pose_extractor.py
from mmpose.apis import init_model, inference_topdown
from mmdet.apis import init_model as init_det
from mmtrack.apis import inference_mot
import cv2
import numpy as np

class MultiPoseExtractor:
    def __init__(self):
        # 检测模型（YOLOX + RTMPose）
        self.det_model = init_det('yolox_l_8x8_300e_coco.py', 'yolox_l_8x8_300e_coco_20211126_140254-ee22ba79.pth', device='cuda')
        self.pose_model = init_model('rtmpose-s_8xb256-420e_coco-256x192.py', 'rtmpose-s_simcc-coco_pt-aic-coco_120e-256x192-f1d8ece0_20230126.pth', device='cuda')
        self.tracker = {}  # 简易 ID 跟踪

    def extract(self, frame):
        result = inference_mot(self.det_model, frame, frame_id=0)
        poses = inference_topdown(self.pose_model, frame, result['track_results'])
        
        output = []
        for res in poses:
            kpts = res.pred_instances.keypoints[0]
            scores = res.pred_instances.keypoint_scores[0]
            track_id = res.track_id
            person_kpts = np.hstack([kpts, scores.reshape(-1,1)])  # (17,3)
            output.append({
                'id': int(track_id),
                'keypoints': person_kpts,
                'bbox': res.bboxes[0].tolist()
            })
        return output
