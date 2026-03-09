import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_size=64, num_layers=2, dropout=0.3):
        super(LSTMModel, self).__init__()
        # Input shape from dataloader is (Batch, 1, Features)
        # We will permute it to (Batch, Features, 1) to treat the feature dimension as a time sequence of length=input_dim
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x is (B, 1, input_dim)
        # Change to (B, input_dim, 1)
        x = x.permute(0, 2, 1)
        
        # LSTM output:
        # out: (Batch, SeqLen, HiddenSize)
        # hn: (NumLayers, Batch, HiddenSize)
        out, (hn, cn) = self.lstm(x)
        
        # Take the output of the last time step
        last_out = out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)
