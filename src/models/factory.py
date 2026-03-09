from .cnn1d import CNN1D
from .cnn2d import CNN2D
from .fusion import FusionModel

def get_model(model_name, input_dim, num_classes, config):
    try:
        model_props = config['model_params'][model_name]
    except KeyError:
        model_props = {'hidden_channels': [64, 128], 'dropout': 0.3}

    if model_name == "cnn1d":
        return CNN1D(input_dim, num_classes, model_props['hidden_channels'], model_props.get('dropout', 0.3))
    elif model_name == "cnn2d":
        return CNN2D(input_dim, num_classes, model_props['hidden_channels'], model_props.get('dropout', 0.3))
    elif model_name == "fusion":
        return FusionModel(input_dim, num_classes, model_props['hidden_channels'], model_props.get('dropout', 0.3))
    else:
        raise ValueError(f"Unknown model type: {model_name}")
