# src/infer.py
import torch
from src.rtmpose_extractor import RTMPoseExtractor
from src.stgcn_model import STGCN
from src.counter import count_jumps
import numpy as np
import yaml

cfg = yaml.safe_load(open('config.yaml'))

# 加载模型
extractor = RTMPoseExtractor()
model = STGCN(num_classes=2)
model.load_state_dict(torch.load('models/checkpoints/stgcn_best.pth'))
model.eval().to('cuda')

# 提取骨骼
video_path = cfg['inference']['video_path']
npy_path = 'data/pose_npy/temp_test.npy'
extractor.extract(video_path, npy_path)

# 分类
data = np.load(npy_path)
tensor = torch.tensor(data).unsqueeze(0).to('cuda')
with torch.no_grad():
    output = model(tensor)
    pred = output.argmax().item()
    prob = torch.softmax(output, 1)[0][pred].item()

print(f"动作: {'跳绳' if pred == 0 else '非跳绳'}, 置信度: {prob:.4f}")

# 计数
if pred == 0:
    count, peaks, signal = count_jumps(data, fps=cfg['inference']['fps'])
    print(f"跳绳次数: {count}")
