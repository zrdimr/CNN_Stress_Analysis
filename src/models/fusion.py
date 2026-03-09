import torch
import torch.nn as nn
import torch.nn.functional as F
from .cnn1d import CNN1D

class FusionModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_channels=[64, 128], dropout=0.3):
        super(FusionModel, self).__init__()
        # 1D CNN Branch (outputs 64 features)
        self.cnn_branch = CNN1D(input_dim, 64, hidden_channels, dropout)
        
        # Dense / MLP Branch
        self.mlp_branch = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # Classification Head
        self.fc_out = nn.Linear(64 + 64, num_classes)
        
    def forward(self, x):
        # x is (B, 1, Seq)
        out_cnn = self.cnn_branch(x)
        
        # Mlp branch prefers (B, Seq)
        out_mlp = self.mlp_branch(x.squeeze(1))
        
        fused = torch.cat([out_cnn, out_mlp], dim=1)
        return self.fc_out(fused)
