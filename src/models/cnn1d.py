import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_channels=[64, 128], dropout=0.3):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels[0])
        self.conv2 = nn.Conv1d(hidden_channels[0], hidden_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels[1])
        self.pool = nn.AdaptiveMaxPool1d(4)
        
        self.fc1 = nn.Linear(hidden_channels[1] * 4, 64)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
