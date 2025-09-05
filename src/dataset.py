# src/dataset.py
"""
跳绳动作数据集定义
支持 ST-GCN 动作分类 和 ScoringNet 评分
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
from typing import List, Tuple, Dict

class JumpRopeDataset(Dataset):
    """
    跳绳动作数据集
    数据格式: (keypoints_sequence, label)
        keypoints_sequence: [T, V, C] -> T=帧数, V=17关键点, C=3(x,y,score)
        label: 动作类别 (0=idle, 1=jumping) 或 评分 (0-100)
    """
    def __init__(self, data_path: str, labels_path: str, seq_len: int = 30, task: str = 'action'):
        """
        Args:
            data_path: 关键点序列路径 (.npy 或 .pkl)
            labels_path: 标签路径
            seq_len: 序列长度（补零或截断）
            task: 'action' 或 'scoring'
        """
        self.seq_len = seq_len
        self.task = task

        # 加载数据
        if data_path.endswith('.npy'):
            self.data = np.load(data_path, allow_pickle=True)  # list of arrays
        elif data_path.endswith('.pkl'):
            with open(data_path, 'rb') as f:
                self.data = pickle.load(f)

        self.labels = np.load(labels_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.data[idx]  # [T, V, C]
        label = self.labels[idx]

        # 截断或补零
        if len(seq) > self.seq_len:
            seq = seq[:self.seq_len]
        elif len(seq) < self.seq_len:
            pad_len = self.seq_len - len(seq)
            pad = np.zeros((pad_len, seq.shape[1], seq.shape[2]))
            seq = np.concatenate([seq, pad], axis=0)

        seq = torch.tensor(seq, dtype=torch.float32)  # [T, V, C]
        label = torch.tensor(label, dtype=torch.float32 if self.task == 'scoring' else torch.long)

        # 归一化 (x,y) 坐标到 [0,1]
        seq[:, :, :2] = seq[:, :, :2] / 1080.0  # 假设 1080p

        return seq, label


# 示例：生成数据的辅助函数
def save_sample_data():
    """生成示例数据（实际应从视频标注中提取）"""
    import numpy as np

    # 模拟 100 个跳绳序列
    data = []
    for _ in range(100):
        T = np.random.randint(25, 35)
        seq = np.random.rand(T, 17, 3).astype(np.float32)
        data.append(seq)

    labels = np.random.randint(0, 2, 100)  # 0=idle, 1=jumping

    np.save('data/action/data.npy', data, allow_pickle=True)
    np.save('data/action/labels.npy', labels)

    print("✅ 示例数据已生成: data/action/data.npy, labels.npy")
