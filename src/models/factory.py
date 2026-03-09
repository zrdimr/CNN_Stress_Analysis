from .cnn1d import CNN1D
from .lstm import LSTMModel
from .cnn_lstm import CNN_LSTM

def get_model(model_name, input_dim, num_classes, config):
    try:
        model_props = config['model_params'].get(model_name, {})
    except KeyError:
        model_props = {}

    if model_name == "cnn1d":
        return CNN1D(input_dim, num_classes, model_props.get('dropout', 0.5))
    elif model_name == "lstm":
        return LSTMModel(input_dim, num_classes, 
                         hidden_size=model_props.get('hidden_size', 64), 
                         num_layers=model_props.get('num_layers', 2), 
                         dropout=model_props.get('dropout', 0.3))
    elif model_name == "cnn_lstm":
        return CNN_LSTM(input_dim, num_classes, 
                        cnn_channels=model_props.get('cnn_channels', [32, 64]), 
                        lstm_hidden=model_props.get('lstm_hidden', 64), 
                        dropout=model_props.get('dropout', 0.3))
    else:
        raise ValueError(f"Unknown model type: {model_name}")
