import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.5):
        super(CNN1D, self).__init__()
        
        # 1. Conv1D layer Filter 512 kernel 3
        self.conv1 = nn.Conv1d(1, 512, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)
        
        # 2. Conv1D layer Filter 256 kernel 3
        self.conv2 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        
        # 3. Conv1D layer Filter 128 kernel 3
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        
        # 4. Conv1D layer Filter 64 kernel 3
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        # Using AdaptiveMaxPool1d at the end of convolutions to ensure the 
        # flattening dimension is fixed regardless of raw input sequence length (e.g. 34 input features)
        self.pool4 = nn.AdaptiveMaxPool1d(2) 
        
        # Fully connected Multi-layer Perceptron
        # Flattened size is 64 channels * 2 spatial dimensions = 128
        self.fc1 = nn.Linear(64 * 2, 1024)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(1024, 512)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(512, 256)
        
        # Output Layer (num_classes neurons instead of 1 Sigmoid since this is multi-class)
        self.out = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Convolutional layers + MaxPooling + ReLU
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC MLP
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        
        return self.out(x)
