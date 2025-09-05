# src/train.py
import os
from src.dataset import JumpRopeDataset
from src.stgcn_model import STGCN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

cfg = yaml.safe_load(open('config.yaml'))

dataset = JumpRopeDataset()
loader = DataLoader(dataset, batch_size=cfg['train']['batch_size'], shuffle=True)
model = STGCN(num_classes=cfg['model']['num_classes']).to(cfg['train']['device'])
opt = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'])
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(cfg['train']['epochs']):
    loss_total = 0
    correct = 0
    for data, target in loader:
        data, target = data.to(cfg['train']['device']), target.to(cfg['train']['device'])
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        opt.step()
        loss_total += loss.item()
        correct += output.argmax(1).eq(target).sum().item()
    acc = 100. * correct / len(dataset)
    print(f"Epoch {epoch+1}: Loss={loss_total:.4f}, Acc={acc:.2f}%")

os.makedirs('models/checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'models/checkpoints/stgcn_best.pth')
