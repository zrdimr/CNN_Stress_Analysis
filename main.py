import yaml
import os
import torch
import warnings
warnings.filterwarnings('ignore')

from src.data.dataset_builder import build_dataloaders
from src.models.factory import get_model
from src.modules.train import train_model
from src.modules.evaluate import evaluate_model
from src.utils.report_generator import generate_professor_level_report

def main():
    print("=== Loading ML Research Configuration ===")
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    balancing_strategies = config['experiments']['balancing_strategies']
    models_to_run = config['experiments']['models']
    
    # Research Matrix registry
    all_results = []
    
    for balancing in balancing_strategies:
        print(f"\n=======================================================")
        print(f"| Preparing Data with Balancing Strategy: [{balancing}] |")
        print(f"=======================================================")
        
        # Build dataset for standard 1D / Fusion (assuming common dim)
        train_loader_1d, test_loader_1d, classes, input_dim = build_dataloaders(config, "cnn1d", balancing)
        train_loader_2d, test_loader_2d, _, _ = build_dataloaders(config, "cnn2d", balancing)
        
        num_classes = len(classes)
        
        for model_name in models_to_run:
            run_name = f"{model_name}_{balancing}"
            print(f"\n--- Initiating Run: {run_name} ---")
            
            # Select proper loader
            current_train_loader = train_loader_2d if model_name == "cnn2d" else train_loader_1d
            current_test_loader  = test_loader_2d if model_name == "cnn2d" else test_loader_1d
            
            # Init Model
            model = get_model(model_name, input_dim, num_classes, config)
            
            # Train
            best_model_path = train_model(model, current_train_loader, current_test_loader, config, run_name)
            
            # Evaluate using best model
            model.load_state_dict(torch.load(best_model_path))
            result_metrics = evaluate_model(model, current_test_loader, classes, config, run_name)
            
            all_results.append(result_metrics)

    print("\n=== Generating Master Research Report ===")
    generate_professor_level_report(all_results, config)
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
