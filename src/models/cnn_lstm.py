import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, num_classes, cnn_channels=[32, 64], lstm_hidden=64, dropout=0.3):
        super(CNN_LSTM, self).__init__()
        # CNN Feature Extractor
        self.conv1 = nn.Conv1d(1, cnn_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels[0])
        self.conv2 = nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels[1])
        
        self.pool = nn.MaxPool1d(2)
        
        # The sequence length will be roughly input_dim // 2 due to pooling
        # The input size to LSTM will be the number of CNN channels
        self.lstm = nn.LSTM(input_size=cnn_channels[1], hidden_size=lstm_hidden, 
                            num_layers=1, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, num_classes)
        
    def forward(self, x):
        # x is (B, 1, SeqLen)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        
        # x is currently (Batch, Channels, SeqLen)
        # LSTM expects (Batch, SeqLen, Features/Channels)
        x = x.permute(0, 2, 1)
        
        # Process through LSTM
        out, (hn, cn) = self.lstm(x)
        
        last_out = out[:, -1, :] # Last sequence output
        last_out = self.dropout(last_out)
        
        return self.fc(last_out)
