# scripts/train.py
"""
统一训练脚本：支持训练 ST-GCN 和 ScoringNet
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import logging
import os
from src.dataset import JumpRopeDataset
from src.stgcn_model import STGCN
from src.scorer import ScoringNet

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if model_name == 'action':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')
            logger.info(f"✅ 模型保存: models/{model_name}_best.pth")

    logger.info(f"✅ 训练完成 | 最佳验证损失: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['action', 'scoring'], help='训练任务')
    parser.add_argument('--data-path', type=str, default='data/action/data.npy')
    parser.add_argument('--labels-path', type=str, default='data/action/labels.npy')
    parser.add_argument('--seq-len', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    # 创建目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # 数据集
    dataset = JumpRopeDataset(args.data_path, args.labels_path, seq_len=args.seq_len, task=args.task)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    logger.info(f"✅ 数据加载完成 | 训练: {len(train_data)} | 验证: {len(val_data)}")

    # 模型
    if args.task == 'action':
        model = STGCN(num_classes=2)
    else:
        model = ScoringNet()

    # 训练
    train_model(model, train_loader, val_loader, args.epochs, args.lr, args.task)


if __name__ == "__main__":
    main()
