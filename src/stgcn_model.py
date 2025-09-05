# src/stgcn_model.py
import torch
import torch.nn as nn

class Graph:
    def __init__(self):
        self.num_nodes = 17
        self.edges = [
            (0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
        ]
        self.adj = self.build_adj()

    def build_adj(self):
        adj = torch.zeros(self.num_nodes, self.num_nodes)
        for i,j in self.edges:
            adj[i,j] = 1
            adj[j,i] = 1
        adj += torch.eye(self.num_nodes)
        deg = adj.sum(1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_norm = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)
        return adj_norm

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, adj):
        super().__init__()
        self.adj = nn.Parameter(adj, requires_grad=False)
        self.conv = nn.Conv2d(in_c, out_c, 1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', x, self.adj)
        x = self.conv(x)
        x = self.bn(x)
        return torch.relu(x)

class STGCN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        graph = Graph()
        self.register_buffer('adj', graph.adj.unsqueeze(0).unsqueeze(0))
        self.layers = nn.Sequential(
            STGCNBlock(3, 64, self.adj.squeeze()),
            STGCNBlock(64, 64, self.adj.squeeze()),
            STGCNBlock(64, 64, self.adj.squeeze()),
            STGCNBlock(64, 128, self.adj.squeeze()),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0,3,1,2)  # (B,T,V,C) -> (B,C,T,V)
        x = self.layers(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return self.fc(x)
