import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import flwr as fl

# --- CRITICAL PATH FIX ---
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if MAIN_DIR not in sys.path:
    sys.path.insert(0, MAIN_DIR)
# -------------------------

from utils.data_manager import get_single_machine_data
from client.partitioner import get_non_iid_partitions, get_iid_partitions
from models.architecture import build_spacecraft_brain

# The ultimate maximum number of columns across all 82 files
MAX_FEATURES = 60  

def pad_features(X, max_features=MAX_FEATURES):
    """
    Pads the 3rd dimension (features) with zeros up to max_features.
    E.g., transforms shape (1000, 100, 25) -> (1000, 100, 60)
    """
    num_windows, window_size, num_features = X.shape
    if num_features >= max_features:
        return X[:, :, :max_features] # Truncate just in case it's over 60
    
    # Pad only the last dimension (features) with zeros
    pad_width = ((0, 0), (0, 0), (0, max_features - num_features))
    return np.pad(X, pad_width, mode='constant', constant_values=0)

def load_client_data(client_id, partition_type="non_iid", alpha=1.0):
    """
    Loads and combines all sensor files assigned to this specific client.
    """
    # 1. Ask the partitioner which channels belong to this client
    if partition_type == "iid":
        # Force exactly 12 clients to match our Non-IID count
        mapping = get_iid_partitions(num_clients=12)
    else:
        mapping = get_non_iid_partitions()
        
    assigned_channels = mapping.get(client_id, [])
    if not assigned_channels:
        raise ValueError(f"Client {client_id} was assigned no channels!")
        
    print(f"📦 Client {client_id} loading {len(assigned_channels)} channels for {partition_type} mode...")

    # 2. Loop through assigned channels, load data, and apply Zero-Padding
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for channel in assigned_channels:
        try:
            # Training Data
            X_tr, y_tr = get_single_machine_data(channel, mode="train", alpha=alpha)
            X_train_list.append(pad_features(X_tr))
            y_train_list.append(y_tr)
            
            # Testing Data
            X_te, y_te = get_single_machine_data(channel, mode="test", alpha=1.0)
            X_test_list.append(pad_features(X_te))
            y_test_list.append(y_te)
        except Exception as e:
            print(f"   ⚠️ Could not load {channel}: {e}")
            
    # Combine all loaded arrays into one giant client dataset
    X_train_combined = np.concatenate(X_train_list, axis=0)
    y_train_combined = np.concatenate(y_train_list, axis=0)
    X_test_combined = np.concatenate(X_test_list, axis=0)
    y_test_combined = np.concatenate(y_test_list, axis=0)
    
    return X_train_combined, y_train_combined, X_test_combined, y_test_combined

# ------------------------------------------------------------------
# 🌸 FLOWER CLIENT CLASS
# ------------------------------------------------------------------
class SpacecraftClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_parameters(self, config):
        """Sends our brain's weights up to the server."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Receives master weights from Server, trains locally, sends updates back."""
        self.model.set_weights(parameters)
        
        # Train on the local spacecraft data
        self.model.fit(self.X_train, self.y_train, epochs=3, batch_size=64, verbose=0)
        
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Takes final exam using local test data and reports score to Server."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        return float(loss), len(self.X_test), {"accuracy": float(accuracy)}

# ------------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Spacecraft Client")
    parser.add_argument("--client_id", type=int, required=True, help="ID of the client (0-11)")
    parser.add_argument("--partition_type", type=str, default="non_iid", choices=["iid", "non_iid"])
    parser.add_argument("--alpha", type=float, default=1.0, help="Amount of training data to use")
    args = parser.parse_args()

    print("="*60)
    print(f"🛰️ STARTING SPACECRAFT CLIENT {args.client_id} ({args.partition_type.upper()}) 🛰️")
    print("="*60)

    # 1. Load data
    X_train, y_train, X_test, y_test = load_client_data(
        client_id=args.client_id, 
        partition_type=args.partition_type, 
        alpha=args.alpha
    )
    
    print(f"✅ Data loaded successfully. Training on {len(X_train)} time-windows.")

    # 2. Build the shared CNN (Window Size=100, Features=60 padded)
    model = build_spacecraft_brain(window_size=100, num_features=MAX_FEATURES)

    # 3. Start Flower Client and connect to local server
    print("📡 Connecting to Space Station Server (127.0.0.1:8080)...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=SpacecraftClient(model, X_train, y_train, X_test, y_test).to_client()
    )