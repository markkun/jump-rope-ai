# scripts/train.py (增强版)
"""
统一训练脚本：支持 ST-GCN 和 ScoringNet
✅ 新增：TensorBoard、学习率调度、早停机制
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import logging
import os
import time
from src.dataset import JumpRopeDataset
from src.stgcn_model import STGCN
from src.scorer import ScoringNet

# 日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss > (self.best_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(model, train_loader, val_loader, epochs, lr, model_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # ✅ TensorBoard
    log_dir = f"logs/train_{model_name}_{time.strftime('%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)

    if model_name == 'action':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    early_stopping = EarlyStopping(patience=15)
    best_loss = float('inf')

    for epoch in range(epochs):
        # 训练
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

        avg_train = train_loss / len(train_loader)

        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)

        # ✅ TensorBoard 写入
        writer.add_scalar('Loss/Train', avg_train, epoch)
        writer.add_scalar('Loss/Val', avg_val, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        # 学习率调度
        scheduler.step(avg_val)

        logger.info(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # 保存最佳模型
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), f'models/{model_name}_best.pth')
            logger.info(f"✅ 模型保存: models/{model_name}_best.pth")

        # 早停
        early_stopping(avg_val)
        if early_stopping.early_stop:
            logger.info("🛑 Early stopping triggered.")
            break

    writer.close()
    logger.info(f"✅ 训练完成 | 最佳验证损失: {best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, choices=['action', 'scoring'])
    parser.add_argument('--data-path', type=str, default='data/action/data.npy')
    parser.add_argument('--labels-path', type=str, default='data/action/labels.npy')
    parser.add_argument('--seq-len', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    dataset = JumpRopeDataset(args.data_path, args.labels_path, seq_len=args.seq_len, task=args.task)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)

    logger.info(f"✅ 数据加载完成 | 训练: {len(train_data)} | 验证: {len(val_data)}")

    model = STGCN(num_classes=2) if args.task == 'action' else ScoringNet()
    train_model(model, train_loader, val_loader, args.epochs, args.lr, args.task)


if __name__ == "__main__":
    main()
