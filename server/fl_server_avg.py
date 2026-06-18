import sys
import os
import flwr as fl

# --- CRITICAL PATH FIX ---
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
# -------------------------

def evaluate_metrics_aggregation_fn(results):
    """Calculates the Global Average Accuracy after all clients finish their exams."""
    if not results:
        return {}
    total_examples = sum([num_examples for num_examples, _ in results])
    weighted_sum = sum([num_examples * m["accuracy"] for num_examples, m in results])
    return {"accuracy": weighted_sum / total_examples}

if __name__ == "__main__":
    print("="*60)
    print("📡 BOOTING UP FED-AVG SPACE STATION SERVER 📡")
    print("="*60)

    # Set up standard FedAvg Strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,            # Require 100% of clients to train
        fraction_evaluate=1.0,       # Require 100% of clients to take the exam
        min_available_clients=12,    # Wait exactly for our 12 partitions
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
    )

    # Open the doors on Port 8080
    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3), # Run for 3 FL rounds
        strategy=strategy,
    )