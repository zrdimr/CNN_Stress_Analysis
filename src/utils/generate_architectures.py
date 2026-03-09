import os
import torch
import warnings
warnings.filterwarnings('ignore')
from torchviz import make_dot

from src.models.cnn1d import CNN1D
from src.models.lstm import LSTMModel
from src.models.cnn_lstm import CNN_LSTM

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_graphs():
    ensure_dir("images")
    
    # 1. CNN1D Graph
    model_cnn1d = CNN1D(input_dim=34, num_classes=3)
    dummy_input = torch.randn(1, 1, 34)
    out_cnn1d = model_cnn1d(dummy_input)
    graph_cnn1d = make_dot(out_cnn1d, params=dict(list(model_cnn1d.named_parameters()) + [('input', dummy_input)]))
    graph_cnn1d.format = "png"
    graph_cnn1d.render("images/CNN1D_Architecture")
    print("Generated CNN1D_Architecture.png")

    # 2. LSTM Graph
    model_lstm = LSTMModel(input_dim=1, num_classes=3)
    dummy_input_lstm = torch.randn(1, 1, 1) # Sequence length 1 so PyTorch doesn't unroll 34 LSTM cells in graphviz
    out_lstm = model_lstm(dummy_input_lstm)
    graph_lstm = make_dot(out_lstm, params=dict(list(model_lstm.named_parameters()) + [('input', dummy_input_lstm)]))
    graph_lstm.format = "png"
    graph_lstm.render("images/LSTM_Architecture")
    print("Generated LSTM_Architecture.png")

    # 3. Hybrid CNN_LSTM Graph
    model_hybrid = CNN_LSTM(input_dim=4, num_classes=3)
    dummy_input_hybrid = torch.randn(1, 1, 4) # Small sequence to survive CNN pool, then unroll ~1 LSTM cell
    out_hybrid = model_hybrid(dummy_input_hybrid)
    graph_hybrid = make_dot(out_hybrid, params=dict(list(model_hybrid.named_parameters()) + [('input', dummy_input_hybrid)]))
    graph_hybrid.format = "png"
    graph_hybrid.render("images/CNN_LSTM_Architecture")
    print("Generated CNN_LSTM_Architecture.png")

if __name__ == "__main__":
    generate_graphs()
