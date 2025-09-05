# src/end2end.py
from .multi_pose_extractor import MultiPoseExtractor
from .stgcn_model import STGCN
from .counter import count_jumps
from .scorer import calculate_jump_score
import torch
import numpy as np
import cv2

class JumpRopePipeline:
    def __init__(self, stgcn_ckpt, device='cuda'):
        self.pose_extractor = MultiPoseExtractor()
        self.device = device
        self.model = STGCN(num_classes=2).to(device)
        self.model.load_state_dict(torch.load(stgcn_ckpt, map_location=device))
        self.model.eval()
        self.buffers = {}  # {id: [T frames]}

    def infer(self, video_path):
        cap = cv2.VideoCapture(video_path)
        results = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret: break

            persons = self.pose_extractor.extract(frame)
            for person in persons:
                pid = person['id']
                kpts = person['keypoints']

                if pid not in self.buffers:
                    self.buffers[pid] = []
                self.buffers[pid].append(kpts)
                self.buffers[pid] = self.buffers[pid][-300:]  # 最多300帧

                if len(self.buffers[pid]) >= 30:
                    data = np.array(self.buffers[pid])
                    count, peaks, signal = count_jumps(data)
                    action_score = self.classify_action(data)
                    score = calculate_jump_score(data, count, peaks) if count > 0 else 0

                    # 更新结果
                    results.append({
                        'id': pid,
                        'frame': frame_id,
                        'count': count,
                        'score': score,
                        'action': 'jumping' if action_score > 0.9 else 'non_jump',
                        'bbox': person['bbox']
                    })
            frame_id += 1

        cap.release()
        return self.aggregate_results(results)

    def classify_action(self, data):
        tensor = torch.tensor(data).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.softmax(output, dim=1)[0]
            return prob[0].item()  # 跳绳概率

    def aggregate_results(self, results):
        final = {}
        for r in results:
            pid = r['id']
            if pid not in final:
                final[pid] = r
            else:
                final[pid]['count'] = max(final[pid]['count'], r['count'])
                final[pid]['score'] = r['score']  # 取最新
        return list(final.values())
