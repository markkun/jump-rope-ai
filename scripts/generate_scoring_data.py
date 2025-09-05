# scripts/generate_scoring_data.py
"""
评分模型训练数据生成器
输入：跳绳视频 + 人工评分（CSV）
输出：features.npy, labels.npy
"""
import os
import cv2
import numpy as np
import pandas as pd
from src.optimized_counter import optimized_count_jumps
from src.optimized_pose_extractor import OptimizedPoseExtractor
import torch

class ScoringDataGenerator:
    def __init__(self, video_dir, label_csv, output_dir='data/scoring'):
        self.video_dir = video_dir
        self.labels = pd.read_csv(label_csv)  # columns: video_name, score
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.pose_extractor = OptimizedPoseExtractor()

    def extract_features(self, data):
        """从 pose_data 提取评分特征"""
        T, V, C = data.shape
        if T < 30: return None

        # 1. 节奏稳定性（std of jump intervals）
        _, peaks, _ = optimized_count_jumps(data)
        rhythm_std = np.std(np.diff(peaks)) if len(peaks) > 1 else 10.0

        # 2. 跳跃高度（髋部移动幅度）
        hip_y = (data[:, 11, 1] + data[:, 12, 1]) / 2
        height_mean = np.mean(np.abs(np.diff(hip_y))) * 100

        # 3. 手臂角度（平均）
        arm_angle = self._arm_angle(data)

        # 4. 持续时间
        duration = len(data)

        # 5. 姿态稳定性（关键点置信度）
        stability = data[:, :, 2].mean()

        # 6. 动作置信度（ST-GCN 输出）
        action_conf = self._action_confidence(data)

        return [rhythm_std, height_mean, arm_angle, duration, stability, action_conf]

    def _arm_angle(self, data):
        left = self._angle(data[:,5,:2], data[:,7,:2], data[:,9,:2])
        right = self._angle(data[:,6,:2], data[:,8,:2], data[:,10,:2])
        return (left + right) / 2

    def _angle(self, a, b, c):
        ba = a - b
        bc = c - b
        cosine = np.sum(ba*bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-6)
        cosine = np.clip(cosine, -1, 1)
        return np.degrees(np.arccos(cosine)).mean()

    def _action_confidence(self, data):
        # 模拟 ST-GCN 推理（此处简化）
        return 0.8 + np.random.rand() * 0.2  # 假设跳绳视频动作置信度较高

    def generate(self):
        features = []
        labels = []

        for _, row in self.labels.iterrows():
            video_path = os.path.join(self.video_dir, row['video_name'])
            if not os.path.exists(video_path):
                continue

            cap = cv2.VideoCapture(video_path)
            pose_buffer = []

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_count += 1

                if frame_count % 5 != 0:  # 每5帧抽一帧
                    continue

                persons = self.pose_extractor.extract(frame)
                for person in persons:
                    pose_buffer.append(person['keypoints'])

            cap.release()

            # 只取前300帧
            pose_data = np.array(pose_buffer[:300])
            feat = self.extract_features(pose_data)
            if feat is not None:
                features.append(feat)
                labels.append(row['score'])

        # 保存
        np.save(os.path.join(self.output_dir, 'features.npy'), np.array(features))
        np.save(os.path.join(self.output_dir, 'labels.npy'), np.array(labels))
        print(f"✅ 生成 {len(features)} 条训练数据")
        return np.array(features), np.array(labels)

# 使用示例
if __name__ == "__main__":
    # 创建标签文件（示例）
    labels_df = pd.DataFrame({
        'video_name': ['jump1.mp4', 'jump2.mp4', 'jump3.mp4'],
        'score': [95.0, 78.0, 88.0]
    })
    labels_df.to_csv('data/labels.csv', index=False)

    generator = ScoringDataGenerator(
        video_dir='data/videos',
        label_csv='data/labels.csv'
    )
    X, y = generator.generate()
