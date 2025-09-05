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
