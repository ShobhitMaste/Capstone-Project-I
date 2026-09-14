import flwr as fl
import torch
import os
import json
import argparse
from typing import Dict, Optional, Tuple, List
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fedtrap.config import get_config
from fedtrap.model import create_mobilenetv2, set_trainable_params
from federated.fedavg import FedAvgWithMetrics

def get_evaluate_fn(model, val_loader, device):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        if val_loader is None:
            return None
            
        set_trainable_params(model, parameters)
        
        model.to(device)
        model.eval()
        
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        # Calculate macro F1
        try:
            from sklearn.metrics import f1_score
            f1 = f1_score(all_labels, all_preds, average='macro')
        except ImportError:
            f1 = 0.0
            
        avg_loss = total_loss / max(1, total)
        accuracy = correct / max(1, total)
        
        # Save global model
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/global_round_{server_round}.pt")
        
        return avg_loss, {"accuracy": accuracy, "f1": f1}
        
    return evaluate

def start_server(config_path=None, model=None, val_loader=None, num_rounds=None, min_clients=2):
    conf = get_config()
    if num_rounds is None:
        num_rounds = conf.get('federated', {}).get('rounds', 10)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model is None:
        num_classes = len(conf.get('classes', []))
        model = create_mobilenetv2(num_classes=num_classes)
        
    eval_fn = get_evaluate_fn(model, val_loader, device)
    
    strategy = FedAvgWithMetrics(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        evaluate_fn=eval_fn,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "accuracy": sum([m * n for n, m in [(c, d["accuracy"]) for c, d in metrics]]) / sum([c for c, _ in metrics]),
        } if metrics else {}
    )
    
    # Store metrics and results
    class ResultsTracker:
        def __init__(self):
            self.history = None
    
    tracker = ResultsTracker()
    
    # Actually run Flower server
    server_address = conf.get('federated', {}).get('server_address', '0.0.0.0:8080')
    history = fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    
    # Log metrics
    strategy.save_communication_log("results/fl_communication.json")
    
    results_dict = {
        "losses_distributed": history.losses_distributed,
        "losses_centralized": history.losses_centralized,
        "metrics_distributed": history.metrics_distributed,
        "metrics_centralized": history.metrics_centralized
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/fl_metrics.json", "w") as f:
        json.dump(results_dict, f, indent=4)
        
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--min_clients', type=int, default=2)
    args = parser.parse_args()
    
    start_server(config_path=args.config, num_rounds=args.rounds, min_clients=args.min_clients)
