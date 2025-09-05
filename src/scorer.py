# src/scorer.py
import numpy as np

def calculate_jump_score(pose_sequence, count, peaks):
    """
    pose_sequence: (T, 17, 3)
    count: 跳跃次数
    peaks: 峰值帧索引
    """
    if count == 0:
        return 0

    # 1. 节奏一致性（间隔标准差）
    intervals = np.diff(peaks)
    rhythm_std = np.std(intervals)
    rhythm_score = max(0, 100 - rhythm_std * 10)

    # 2. 跳跃高度（平均位移）
    hip_y = (pose_sequence[:,11,1] + pose_sequence[:,12,1]) / 2
    jump_heights = [hip_y[p] for p in peaks]
    height_mean = np.mean(np.diff([hip_y.min()] + jump_heights))
    height_score = min(100, height_mean * 50)

    # 3. 手臂角度（肘部弯曲检测）
    left_arm_angle = angle_between(pose_sequence[:,5,:2], pose_sequence[:,7,:2], pose_sequence[:,9,:2])
    right_arm_angle = angle_between(pose_sequence[:,6,:2], pose_sequence[:,8,:2], pose_sequence[:,10,:2])
    arm_angle_mean = (left_arm_angle + right_arm_angle) / 2
    arm_score = 100 if 140 < arm_angle_mean < 180 else 50  # 伸直为佳

    # 4. 综合评分
    total = (rhythm_score * 0.4 + height_score * 0.3 + arm_score * 0.3)
    return max(0, min(100, int(total)))

def angle_between(a, b, c):
    ba = a - b
    bc = c - b
    cosine_angle = np.einsum('ij,ij->i', ba, bc) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1))
    cosine_angle = np.clip(cosine_angle, -1, 1)
    return np.degrees(np.arccos(cosine_angle)).mean()

# src/scorer.py
import torch
import torch.nn as nn

class ScoringNet(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        score = self.fc(x)  # [0,1]
        return self.sigmoid(score) * 100  # [0,100]

# 训练数据特征：节奏std, 高度mean, 手臂角, 持续时间, 稳定性...
# 可收集人工评分数据进行微调
