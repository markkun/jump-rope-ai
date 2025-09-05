# src/pipeline.py
from .optimized_pose_extractor import OptimizedPoseExtractor
from .stgcn_model import STGCN
from .optimized_counter import optimized_count_jumps
from .scorer import ScoringNet
import torch
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JumpRopePipeline:
    def __init__(self, stgcn_ckpt, scoring_ckpt=None, device='cuda', max_tracks=10):
        self.device = device
        self.max_tracks = max_tracks
        
        # 模型
        self.pose_extractor = OptimizedPoseExtractor(device=device)
        self.action_model = STGCN(num_classes=2).to(device)
        self.action_model.load_state_dict(torch.load(stgcn_ckpt, map_location=device))
        self.action_model.eval()
        
        self.scoring_model = ScoringNet().to(device)
        if scoring_ckpt:
            self.scoring_model.load_state_dict(torch.load(scoring_ckpt, map_location=device))
        self.scoring_model.eval()

        # 跟踪缓冲
        self.keypoint_buffer = defaultdict(list)  # id -> [frames]
        self.result_buffer = defaultdict(dict)    # id -> {count, score, ...}

    @torch.no_grad()
    def infer_frame(self, frame):
        persons = self.pose_extractor.extract(frame)
        results = []

        for person in persons:
            pid = person['id']
            kpts = person['keypoints']

            # 更新缓冲
            self.keypoint_buffer[pid].append(kpts)
            self.keypoint_buffer[pid] = self.keypoint_buffer[pid][-300:]  # 最大300帧

            if len(self.keypoint_buffer[pid]) >= 30:
                data = np.array(self.keypoint_buffer[pid])

                # 计数
                count, peaks, signal = optimized_count_jumps(data)

                # 动作识别
                tensor = torch.tensor(data).unsqueeze(0).to(self.device)
                action_prob = torch.softmax(self.action_model(tensor), 1)[0, 0].item()

                # 评分（可选）
                score = self._compute_score(data, count, peaks, action_prob)

                result = {
                    'id': int(pid),
                    'count': int(count),
                    'score': float(score),
                    'action_confidence': float(action_prob),
                    'action': 'jumping' if action_prob > 0.85 else 'non_jump',
                    'bbox': [float(x) for x in person['bbox']],
                    'timestamp': len(self.keypoint_buffer[pid])
                }
                self.result_buffer[pid] = result
                results.append(result)

        # 清理过期 ID
        current_ids = {p['id'] for p in persons}
        for pid in list(self.keypoint_buffer.keys()):
            if pid not in current_ids:
                self.keypoint_buffer[pid] = self.keypoint_buffer[pid][-50:]  # 缓存50帧

        return results

    def _compute_score(self, data, count, peaks, action_prob):
        if count == 0:
            return 0.0
        
        # 提取评分特征
        hip_y = (data[:,11,1] + data[:,12,1])/2
        rhythm_std = np.std(np.diff(peaks)) if len(peaks) > 1 else 0
        height_mean = np.mean(np.abs(np.diff(hip_y))) * 100
        duration = len(data)
        
        # 手臂角度
        arm_angle = self._arm_angle(data)
        
        # 稳定性（关键点置信度）
        stability = data[:, :, 2].mean()
        
        features = np.array([rhythm_std, height_mean, arm_angle, duration, stability, action_prob])
        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.scoring_model(features).cpu().item()
        return max(0, min(100, score))

    def _arm_angle(self, data):
        # 计算平均手臂角度
        left = self._angle(data[:,5,:2], data[:,7,:2], data[:,9,:2])
        right = self._angle(data[:,6,:2], data[:,8,:2], data[:,10,:2])
        return (left + right) / 2

    def _angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.sum(ba*bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
        cosine = np.clip(cosine, -1, 1)
        return np.degrees(np.arccos(cosine)).mean()
