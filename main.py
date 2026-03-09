import yaml
import os
import torch
import warnings
import argparse
import json
warnings.filterwarnings('ignore')

from src.data.dataset_builder import build_dataloaders
from src.models.factory import get_model
from src.modules.train import train_model
from src.modules.evaluate import evaluate_model
from src.utils.report_generator import generate_professor_level_report

def run_single_experiment(model_name, balancing, config):
    print(f"\n=======================================================")
    print(f"| Run: [{model_name}] | Balancing: [{balancing}] |")
    print(f"=======================================================")
    
    train_loader, test_loader, classes, input_dim = build_dataloaders(config, model_name, balancing)
    num_classes = len(classes)
    
    run_name = f"{model_name}_{balancing}"
    model = get_model(model_name, input_dim, num_classes, config)
    
    best_model_path = train_model(model, train_loader, test_loader, config, run_name)
    
    model.load_state_dict(torch.load(best_model_path))
    result_metrics = evaluate_model(model, test_loader, classes, config, run_name)
    
    # Save isolated results JSON for merging later in distributed CI/CD
    os.makedirs('reports', exist_ok=True)
    with open(f"reports/{run_name}_result.json", "w") as f:
        json.dump(result_metrics, f)
        
    return result_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Specific model to run (e.g. cnn1d, lstm, cnn_lstm)")
    parser.add_argument("--balancing", type=str, help="Specific balancing (e.g. none, smote, adasyn)")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate existing JSON reports into Professor Report")
    args = parser.parse_args()
    
    print("=== Loading ML Research Configuration ===")
    with open('configs/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    if args.aggregate:
        print("=== Aggregating Distributed Reports ===")
        all_results = []
        import glob
        for file in glob.glob("reports/*_result.json"):
            with open(file, 'r') as f:
                all_results.append(json.load(f))
        generate_professor_level_report(all_results, config)
        return

    if args.model and args.balancing:
        # Run isolated target
        run_single_experiment(args.model, args.balancing, config)
    else:
        # Run sequential loop locally
        all_results = []
        for balancing in config['experiments']['balancing_strategies']:
            for model_name in config['experiments']['models']:
                res = run_single_experiment(model_name, balancing, config)
                all_results.append(res)
        
        print("\n=== Generating Master Research Report ===")
        generate_professor_level_report(all_results, config)
    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    main()
