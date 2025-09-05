# src/dataset.py
import torch
import numpy as np
import os
import random
import yaml

def load_config():
    with open('../config.yaml', 'r') as f:
        return yaml.safe_load(f)

cfg = load_config()

class JumpRopeDataset(torch.utils.data.Dataset):
    def __init__(self, transform=True):
        self.transform = transform
        self.samples = []
        self.max_frames = cfg['model']['max_frames']
        label_map = cfg['data']['label_map']
        for label_name, label_idx in label_map.items():
            path = os.path.join(cfg['data']['pose_dir'], label_name)
            for f in os.listdir(path):
                if f.endswith('.npy'):
                    self.samples.append((os.path.join(path, f), label_idx))

    def __len__(self): return len(self.samples)

    def augment(self, data):
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02, data[:,:2].shape)
            data[:,:2] += noise
        if random.random() < 0.5:
            T = len(data)
            start = random.randint(0, T//4)
            end = random.randint(3*T//4, T)
            data = data[start:end]
        if random.random() < 0.3:
            f, k = random.randint(0, data.shape[0]-1), random.randint(0,16)
            data[f,k,:] = 0
        return data

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        data = np.load(npy_path)

        if self.transform:
            data = self.augment(data)

        T = data.shape[0]
        if T > self.max_frames:
            data = data[:self.max_frames]
        else:
            pad = np.zeros((self.max_frames - T, 17, 3))
            data = np.vstack([data, pad])

        return torch.tensor(data, dtype=torch.float32), label
